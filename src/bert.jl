using Knet
import Base: length, iterate
using Random
using CSV
using PyCall
using SpecialFunctions: erf

const BERTVOCPATH = joinpath(DATADIR, "bert", "bert-base-uncased-vocab.txt")

gelu(x) = x .* 0.5 .* (1.0 .+ erf.(x ./ sqrt(2.0)))
std2(a, μ, ϵ) = sqrt.(Knet.mean(abs2.(a .- μ), dims=1) .+ ϵ)

function matmul23(a, b)
    a = a * reshape(b, size(b)[1], :)
    return reshape(a, :, size(b)[2:end]...)
end


abstract type Layer end

mutable struct Embedding <: Layer
    w
end

Embedding(vocabsize::Int,embed::Int; atype=Array{Float32}) = Embedding(param(embed,vocabsize, atype=atype))

function (e::Embedding)(x)
    e.w[:,x]
end

mutable struct Linear <: Layer
    w
    b
end

Linear(input_size::Int, output_size::Int; atype=Array{Float32}) = Linear(param(output_size, input_size, atype=atype), param0(output_size, atype=atype))

function (l::Linear)(x)
    return l.w * x .+ l.b
end

mutable struct Linear3D <: Layer
    w
    b
end

Linear3D(input_size::Int, output_size::Int; atype=Array{Float32}) = Linear3D(param(output_size, input_size, atype=atype), param0(output_size, atype=atype))

function (l::Linear3D)(x)
    return matmul23(l.w, x) .+ l.b
end

# Absolutely no difference between Dense and Linear! Except one has dropout and activation function.
mutable struct Dense <: Layer
    linear
    pdrop
    func
end

function Dense(input_size::Int, output_size::Int; pdrop=0.0, func=identity, atype=Array{Float32}, threeD=false)
    if threeD
        return Dense(Linear3D(input_size, output_size, atype=atype), pdrop, func)
    else
        return Dense(Linear(input_size, output_size, atype=atype), pdrop, func)
    end
end

function (a::Dense)(x)
    return a.func.(dropout(a.linear(x), a.pdrop))
end

mutable struct LayerNormalization <: Layer
    γ
    β
    ϵ
end

LayerNormalization(hidden_size::Int; epsilon=1e-12, atype=Array{Float32}) = LayerNormalization(Param(atype(ones(hidden_size))), param0(hidden_size, atype=atype), epsilon)

function (n::LayerNormalization)(x)
    μ = Knet.mean(x, dims=1)
    x = (x .- μ) ./ std2(x, μ, n.ϵ) # corrected=false for n
    return n.γ .* x .+ n.β
end

mutable struct EmbedLayer <: Layer
    wordpiece::Embedding
    positional::Embedding
#    segment::SegmentEmbedding
    segment::Embedding
    layer_norm::LayerNormalization
    seq_len::Int
    pdrop
end

function EmbedLayer(config)
    wordpiece = Embedding(config.vocab_size, config.embed_size, atype=config.atype)
    positional = Embedding(config.max_seq_len, config.embed_size, atype=config.atype)
    #segment = SegmentEmbedding(config.num_segment, config.embed_size, atype=config.atype)
    segment = Embedding(config.num_segment, config.embed_size, atype=config.atype)
    layer_norm = LayerNormalization(config.embed_size, atype=config.atype)
    return EmbedLayer(wordpiece, positional, segment, layer_norm, config.seq_len, config.pdrop)
end

function (e::EmbedLayer)(x, segment_ids) # segment_ids are SxB, containing 1 or 2, or 0 in case of pads.
    x = e.wordpiece(x)
    positions = zeros(Int64, e.seq_len, size(x,3)) .+ collect(1:e.seq_len) # size(x,3) is batchsizee. Resulting matrix is SxB
    x = x .+ e.positional(positions)
    #x .+= reshape(hcat(e.segment.(segment_ids)...), (:, size(segment_ids,1),size(segment_ids,2)))
    x = x .+ e.segment(segment_ids)
    x = e.layer_norm(x)
    return dropout(x, e.pdrop)
end

function divide_to_heads(x, num_heads, head_size, seq_len)
    x = reshape(x, (head_size, num_heads, seq_len, :))
    x = permutedims(x, (1,3,2,4))
    return reshape(x, (head_size, seq_len, :)) # Reshape to 3D so bmm can handle it.
end

mutable struct SelfAttention <: Layer
    query::Linear3D # N*H x E
    key::Linear3D
    value::Linear3D
    linear::Linear3D
    num_heads::Int
    seq_len::Int
    embed_size::Int
    head_size::Int
    head_size_sqrt::Int
    attention_pdrop
    pdrop
end

function SelfAttention(config)
    config.embed_size % config.num_heads != 0 && throw("Embed size should be divisible by number of heads!")
    head_size = Int(config.embed_size / config.num_heads)
    head_size_sqrt = Int(sqrt(head_size))
    head_size_sqrt * head_size_sqrt != head_size && throw("Square root of head size should be an integer!")
    query = Linear3D(config.embed_size, head_size*config.num_heads, atype=config.atype) # H*N is always equal to E
    key = Linear3D(config.embed_size, head_size*config.num_heads, atype=config.atype)
    value = Linear3D(config.embed_size, head_size*config.num_heads, atype=config.atype)
    linear = Linear3D(config.embed_size, config.embed_size, atype=config.atype)
    return SelfAttention(query, key, value, linear, config.num_heads, config.seq_len, config.embed_size, head_size, head_size_sqrt, config.attention_pdrop, config.pdrop)
end

function (s::SelfAttention)(x, attention_mask)
    # We make all the batchsize ones colon, in case of batches smaller than batchsize.
    # x is ExSxB
    query = divide_to_heads(s.query(x), s.num_heads, s.head_size, s.seq_len) # H x S x N*B
    key = divide_to_heads(s.key(x), s.num_heads, s.head_size, s.seq_len)
    value = divide_to_heads(s.value(x), s.num_heads, s.head_size, s.seq_len)
    
    # Scaled Dot Product Attention
    query = bmm(permutedims(key, (2,1,3)), query)
    query = query ./ s.head_size_sqrt # Scale down. I init this value to avoid taking sqrt every forward operation.
    # Masking. First reshape to 4d, then add mask, then reshape back to 3d.
    query = reshape(reshape(query, (s.seq_len, s.seq_len, s.num_heads, :)) .+ attention_mask, (s.seq_len, s.seq_len, :))

    query = Knet.softmax(query, dims=1)
    query = dropout(query, s.attention_pdrop)
    query = bmm(value, query)
    query = permutedims(reshape(query, (s.head_size, s.seq_len, s.num_heads, :)), (1,3,2,4))
    
    query = reshape(query, (s.embed_size, s.seq_len, :)) # Concat
    return dropout(s.linear(query), s.pdrop) # Linear transformation at the end
    # In pytorch version dropout is after layer_norm!
end

mutable struct FeedForward <: Layer
    dense::Dense
    linear::Linear3D
    pdrop
end

function FeedForward(config)
    dense = Dense(config.embed_size, config.ff_hidden_size, func=eval(Meta.parse(config.func)), atype=config.atype, threeD=true)
    linear = Linear3D(config.ff_hidden_size, config.embed_size, atype=config.atype)
    return FeedForward(dense, linear, config.pdrop)
end

function (f::FeedForward)(x)
    x = f.dense(x)
    return dropout(f.linear(x), f.pdrop)
end

mutable struct Encoder <: Layer
    self_attention::SelfAttention
    layer_norm1::LayerNormalization
    feed_forward::FeedForward
    layer_norm2::LayerNormalization
end

function Encoder(config)
    return Encoder(SelfAttention(config), LayerNormalization(config.embed_size, atype=config.atype), FeedForward(config), LayerNormalization(config.embed_size, atype=config.atype))
end

function (e::Encoder)(x, attention_mask)
    x = e.layer_norm1(x .+ e.self_attention(x, attention_mask))
    return e.layer_norm2(x .+ e.feed_forward(x))
end

mutable struct Bert <: Layer
    embed_layer::EmbedLayer
    encoder_stack
    atype
end

function Bert(config)
    embed_layer = EmbedLayer(config)
    encoder_stack = Encoder[]
    for _ in 1:config.num_encoder
        push!(encoder_stack, Encoder(config))
    end
    return Bert(embed_layer, encoder_stack, config.atype)
end

# x and segment_ids are SxB integers
function (b::Bert)(x, segment_ids; attention_mask=nothing, extractlayer=11)
    # Init attention_mask if it's not given
    attention_mask = attention_mask == nothing ? ones(size(x)) : attention_mask
    attention_mask = reshape(attention_mask, (size(attention_mask,1), 1, 1, size(attention_mask,2))) # Make it 4d
    attention_mask = (1 .- attention_mask) .* -10000.0 # If integer was 0, now it is masking. ones(size(attention_mask))
    attention_mask = b.atype(attention_mask)

    x = b.embed_layer(x, segment_ids)

    numencoder = length(b.encoder_stack)
    extractlayer = extractlayer >= numencoder ? numencoder : extractlayer

    for i in 1:extractlayer
        x = b.encoder_stack[i](x, attention_mask)
    end

    return x
end

mutable struct Pooler <: Layer
    linear::Linear
end

Pooler(embed_size::Int; atype=Array{Float32}) = Pooler(Linear(embed_size, embed_size, atype=atype))

function (p::Pooler)(x)
    # TODO :
    # Gave up on getindex function for 3D matrices because I could not figure out how to write setindex! for backprop
#     x = reshape(x, :, size(x,3))
#    return tanh.(p.linear(x[:,1,:])) # Use only CLS token. Returns ExB
    return tanh.(p.linear(reshape(x, :, size(x,3))[1:size(x,1),:]))
end

mutable struct NSPHead <: Layer
    linear::Linear
end

NSPHead(embed_size::Int; atype=Array{Float32}) = NSPHead(Linear(embed_size, 2, atype=atype))

(n::NSPHead)(x) = n.linear(x)

mutable struct MLMHead <: Layer
    dense::Dense
    layer_norm::LayerNormalization
    linear::Linear3D
end

function MLMHead(config, embedding_matrix)
    dense = Dense(config.embed_size, config.embed_size, func=eval(Meta.parse(config.func)), pdrop=0.0, atype=config.atype, threeD=true)
    layer_norm = LayerNormalization(config.embed_size, atype=config.atype)
    linear = Linear3D(config.embed_size, config.vocab_size, atype=config.atype)
    # TODO : Do this a shared weight
    #linear.w = permutedims(embedding_matrix, (2,1))
    return MLMHead(dense, layer_norm, linear)
end

function (m::MLMHead)(x)
    x = m.dense(x)
    x = m.layer_norm(x)
    return m.linear(x)
end

mutable struct BertPreTraining <: Layer
    bert::Bert
    pooler::Pooler
    nsp::NSPHead
    mlm::MLMHead
end

function BertPreTraining(config)
    bert = Bert(config)
    pooler = Pooler(config.embed_size, atype=config.atype)
    nsp = NSPHead(config.embed_size, atype=config.atype)
    mlm = MLMHead(config, bert.embed_layer.wordpiece.w) # TODO : Dont forget about embedding matrix
    return BertPreTraining(bert, pooler, nsp, mlm)
end

# We do not need a predictor, since this is only for pretraining
function (b::BertPreTraining)(x, segment_ids, mlm_labels, nsp_labels; attention_mask=nothing) # mlm_labels are SxB, so we just flatten them.
    x = b.bert(x, segment_ids, attention_mask=attention_mask)
    nsp_preds = b.nsp(b.pooler(x)) # 2xB
    mlm_preds = b.mlm(x) # VxSxB
    mlm_preds = reshape(mlm_preds, size(mlm_preds, 1), :) # VxS*B
    nsp_loss = nll(nsp_preds, nsp_labels)
    mlm_labels = reshape(mlm_labels, :) # S*B
    mlm_loss = nll(mlm_preds[:,mlm_labels.!=-1], mlm_labels[mlm_labels.!=-1])
    return mlm_loss + nsp_loss
end

function (b::BertPreTraining)(dtrn)
    lvals = []
    for (x, attention_mask, segment_ids, mlm_labels, nsp_labels) in dtrn
        push!(lvals, b(x, segment_ids, mlm_labels, nsp_labels, attention_mask=attention_mask))
    end
    return Knet.mean(lvals)
end

mutable struct BertClassification <: Layer
    bert::Bert
    pooler::Pooler
    linear::Linear
    pdrop
end

function BertClassification(config, num_of_classes)
    bert = Bert(config)
    pooler = Pooler(config.embed_size, atype=config.atype)
    linear = Linear(config.embed_size, num_of_classes, atype=config.atype)
    return BertClassification(bert, pooler, linear, config.pdrop)
end

function (b::BertClassification)(x, segment_ids; attention_mask=nothing)
    x = b.bert(x, segment_ids, attention_mask=attention_mask)
    x = dropout(b.pooler(x), b.pdrop) # 2xB
    return b.linear(x)
end

function (b::BertClassification)(x, segment_ids, y; attention_mask=nothing)
    return nll(b(x, segment_ids, attention_mask=attention_mask), y)
end

function (b::BertClassification)(dtrn)
    lvals = []
    for (x, attention_mask, segment_ids, y) in dtrn
        push!(lvals, b(x, segment_ids, y, attention_mask=attention_mask))
    end
    return Knet.mean(lvals)
end

mutable struct BertConfig
    embed_size::Int
    vocab_size::Int
    ff_hidden_size::Int
    max_seq_len::Int
    seq_len::Int
    num_segment::Int
    num_heads::Int
    num_encoder::Int
    batchsize::Int
    atype
    pdrop
    attention_pdrop
    func
end

function load_from_torch_base(model, num_encoder, atype, torch_model)
    # Embed Layer
    model.bert.embed_layer.wordpiece.w = Param(atype(permutedims(torch_model["bert.embeddings.word_embeddings.weight"][:cpu]()[:numpy](), (2,1))))
    model.bert.embed_layer.positional.w = Param(atype(permutedims(torch_model["bert.embeddings.position_embeddings.weight"][:cpu]()[:numpy](), (2,1))))
    model.bert.embed_layer.segment.w = Param(atype(permutedims(torch_model["bert.embeddings.token_type_embeddings.weight"][:cpu]()[:numpy](), (2,1))))
    model.bert.embed_layer.layer_norm.γ = Param(atype(torch_model["bert.embeddings.LayerNorm.weight"][:cpu]()[:numpy]()))
    model.bert.embed_layer.layer_norm.β = Param(atype(torch_model["bert.embeddings.LayerNorm.bias"][:cpu]()[:numpy]()))
    
    # Encoder Stack
    for i in 1:num_encoder
        model.bert.encoder_stack[i].self_attention.query.w = Param(atype(torch_model["bert.encoder.layer.$(i-1).attention.self.query.weight"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].self_attention.query.b = Param(atype(torch_model["bert.encoder.layer.$(i-1).attention.self.query.bias"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].self_attention.key.w = Param(atype(torch_model["bert.encoder.layer.$(i-1).attention.self.key.weight"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].self_attention.key.b = Param(atype(torch_model["bert.encoder.layer.$(i-1).attention.self.key.bias"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].self_attention.value.w = Param(atype(torch_model["bert.encoder.layer.$(i-1).attention.self.value.weight"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].self_attention.value.b = Param(atype(torch_model["bert.encoder.layer.$(i-1).attention.self.value.bias"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].self_attention.linear.w = Param(atype(torch_model["bert.encoder.layer.$(i-1).attention.output.dense.weight"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].self_attention.linear.b = Param(atype(torch_model["bert.encoder.layer.$(i-1).attention.output.dense.bias"][:cpu]()[:numpy]()))
        
        model.bert.encoder_stack[i].layer_norm1.γ = Param(atype(torch_model["bert.encoder.layer.$(i-1).attention.output.LayerNorm.weight"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].layer_norm1.β = Param(atype(torch_model["bert.encoder.layer.$(i-1).attention.output.LayerNorm.bias"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].feed_forward.dense.linear.w = Param(atype(torch_model["bert.encoder.layer.$(i-1).intermediate.dense.weight"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].feed_forward.dense.linear.b = Param(atype(torch_model["bert.encoder.layer.$(i-1).intermediate.dense.bias"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].feed_forward.linear.w = Param(atype(torch_model["bert.encoder.layer.$(i-1).output.dense.weight"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].feed_forward.linear.b = Param(atype(torch_model["bert.encoder.layer.$(i-1).output.dense.bias"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].layer_norm2.γ = Param(atype(torch_model["bert.encoder.layer.$(i-1).output.LayerNorm.weight"][:cpu]()[:numpy]()))
        model.bert.encoder_stack[i].layer_norm2.β = Param(atype(torch_model["bert.encoder.layer.$(i-1).output.LayerNorm.bias"][:cpu]()[:numpy]()))
    end
    
    # Pooler
    model.pooler.linear.w = Param(atype(torch_model["bert.pooler.dense.weight"][:cpu]()[:numpy]()))
    model.pooler.linear.b = Param(atype(torch_model["bert.pooler.dense.bias"][:cpu]()[:numpy]()))
    
    return model
end

function load_from_torch_pretraining(model, num_encoder, atype, torch_model)
    model = load_from_torch_base(model, num_encoder, atype, torch_model)
    
    # NSP Head
    model.nsp.linear.w = Param(atype(torch_model["cls.seq_relationship.weight"][:cpu]()[:numpy]()))
    model.nsp.linear.b = Param(atype(torch_model["cls.seq_relationship.bias"][:cpu]()[:numpy]()))
    
    # MLM Head.
    model.mlm.dense.linear.w = Param(atype(torch_model["cls.predictions.transform.dense.weight"][:cpu]()[:numpy]()))
    model.mlm.dense.linear.b = Param(atype(torch_model["cls.predictions.transform.dense.bias"][:cpu]()[:numpy]()))
    model.mlm.layer_norm.γ = Param(atype(torch_model["cls.predictions.transform.LayerNorm.gamma"][:cpu]()[:numpy]()))
    model.mlm.layer_norm.β = Param(atype(torch_model["cls.predictions.transform.LayerNorm.beta"][:cpu]()[:numpy]()))
    model.mlm.linear.w = Param(atype(torch_model["cls.predictions.decoder.weight"][:cpu]()[:numpy]()))
    model.mlm.linear.b = Param(atype(torch_model["cls.predictions.bias"][:cpu]()[:numpy]()))
    
    return model
end

function load_from_torch_classification(model, num_encoder, atype, torch_model)
    model = load_from_torch_base(model, num_encoder, atype, torch_model)
    
    model.linear.w = Param(atype(torch_model["classifier.weight"][:cpu]()[:numpy]()))
    model.linear.b = Param(atype(torch_model["classifier.bias"][:cpu]()[:numpy]()))
    
    return model
end

function bert(; atype=nothing)
    if atype == nothing
        atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}
    end

    config = BertConfig(768, 30522, 3072, 512, 64, 2, 12, 12, 8,
                        atype, 0.1, 0.1, "gelu")
    model = BertClassification(config, 2)
    return model, config
    #@pyimport torch
    #torch_model = torch.load(joinpath(DATADIR, "bert", "model.pt"))
    #model = load_from_torch_base(model, config.num_encoder, config.atype, torch_model)
    #vocab = initalizevocab(joinpath(DATADIR, "bert", "bert-base-uncased-vocab.txt"))
    #input_ids, input_masks, segment_ids = read_and_process(data[fname], vocab)
    #features = [model.bert(input_ids[i], segmentids[i]; attention_mask=input_masks[i]), for i=1:length(input_ids)]
end

function PretrainedBert()
    model, config = AutoML.bert()
    torch = pyimport("torch")
    torch_model = torch.load(joinpath(DATADIR, "bert", "model.pt"))
    model = load_from_torch_base(model, config.num_encoder, config.atype, torch_model)
end

function wordpiece_tokenize(token, dict)
    # This is a longest-match-first algorithm.
    out_tokens = []
    start = 1
    while start <= length(token)
        finish = length(token)
        final_token = ""
        for i in finish:-1:start
            # String Indexing Error for an unknown reason. Might be because of unicode chars.
            tkn = try
                start == 1 ? token[start:i] : string("##", token[start:i])
            catch
                ""
            end
            if tkn in keys(dict)
                final_token = tkn
                finish = i
                break
            end
        end
        if final_token == "" # if there is no match at all, assign unk token
            return ["[UNK]"]
        end
        push!(out_tokens, final_token)
        start = finish + 1
    end
    return out_tokens
end

function process_punc(tokens)
    out_tokens = []
    for token in tokens
        out = []
        str = ""
        for (i, char) in enumerate(token)
            if ispunct(char)
                str != "" && push!(out, str)
                str = ""
                push!(out, string(char))
            else
                str = string(str, char)
            end
        end
        str != "" && push!(out, str)
        append!(out_tokens, out)
    end
    return out_tokens
end

function bert_tokenize(text, dict; lower_case=true)
    text = strip(text)
    text == "" && return []
    if lower_case
        text = lowercase(text)
    end
    tokens = split(text)
    tokens = process_punc(tokens)
    out_tokens = []
    for token in tokens
        append!(out_tokens, wordpiece_tokenize(token, dict))
    end
    return out_tokens
end

function initializevocab(vocpath)
    token2int = Dict()
    f = open(vocpath) do file
        lines = readlines(file)
        for (i,line) in enumerate(lines)
            token2int[line] = i
        end
    end
    token2int
end

function int2token(token2int)
    dict = Dict(value => key for (key, value) in token2int)
end

function convert_to_int_array(text, dict; lower_case=true)
    tokens = bert_tokenize(text, dict, lower_case=lower_case)
    out = Int[]
    for token in tokens
        if token in keys(dict)
            push!(out, dict[token])
        else
            push!(out, dict["[UNK]"])
        end
    end
    return out
end

function read_and_process(data, dict; lower_case=true)
    x = [convert_to_int_array(doc, dict, lower_case=lower_case) for doc in data]
    
    # Padding to maximum
#     max_seq = findmax(length.(x))[1]
#     for i in 1:length(x)
#         append!(x[i], fill(1, max_seq - length(x[i]))) # 1 is for "[PAD]"
#     end
    
    return x
end

function preprocessbert(x; lendoc=64)
    input_mask = []
    input_ids = []
    segment_ids = []
    
    for i in 1:length(x)
        x2 = x[i]
        if length(x[i]) >= lendoc
            x2 = x2[1:lendoc]
            mask = Array{Int64}(ones(lendoc))
        else
            mask = Array{Int64}(ones(length(x2)))
            append!(x2, fill(1, lendoc - length(x2))) # 1 is for "[PAD]"
            append!(mask, fill(0, lendoc - length(mask))) # 0's vanish with masking operation
        end
        push!(input_ids, x2)
        push!(input_mask, mask)
        push!(segment_ids, Array{Int64}(ones(lendoc)))
    end
    input_ids, input_mask, segment_ids
end

function loadpretrainedbert(; loadpath=nothing)
    loadpath = loadpath != nothing ? loadpath :
        joinpath(DATADIR, "bert", "bert.jld2")
    model, config = bert()
    model.bert = load(loadpath, "bert")
    model.pooler = load(loadpath, "pooler")
    model.linear = load(loadpath, "linear")
    model.pdrop = load(loadpath, "pdrop")
    return model
end
