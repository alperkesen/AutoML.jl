using Knet: Knet, gpu, conv4, value

include(Knet.dir("data","imagenet.jl"))

mutable struct ResNet
    w
    ms
    avgimg
    atype
end

function ResNet(;name="imagenet-resnet-101-dag", atype=nothing)
    if atype == nothing
        atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}
    end

    model = matconvnet(name)

    avgimg = model["meta"]["normalization"]["averageImage"]
    avgimg = convert(Array{Float32}, avgimg)

    w, ms = get_params(model["params"], atype)
    ResNet(w, ms, avgimg, atype)
end

function (r::ResNet)(path::String)
    img = imgdata(path, r.avgimg)
    x = convert(r.atype, img)
    resnet101(r.w, x, r.ms)
end

function resnet(path)
    atype = gpu()>=0 ? KnetArray{Float64} : Array{Float64}
    modelname = "imagenet-resnet-101-dag"
    model = matconvnet(modelname)
    avgimg = model["meta"]["normalization"]["averageImage"]
    avgimg = convert(Array{Float64}, avgimg)
    img = imgdata(path, avgimg)
    x = convert(atype, img)
    
    w, ms = get_params(model["params"], atype)
    return resnet101(w,x,ms)
end

function resnet101(w,x,ms; mode=1)
    # layer 1
    conv1 = reslayerx1(w[1:3],x,ms; padding=3, stride=2, mode=mode)
    pool1 = pool(conv1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[4:33], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[34:72], r2, ms; mode=mode)
    r4 = reslayerx5(w[73:282], r3, ms; mode=mode)
    r5 = reslayerx5(w[283:312], r4, ms; mode=mode)

    windowsize = size(r5, 1)
    # fully connected layer
    pool5  = pool(r5; stride=1, window=windowsize, mode=2)
    return mat(pool5)
    #fc1000 = w[313] * mat(pool5) .+ w[314]
end

function batchnorm(w, x, ms; mode=1, epsilon=1e-5)
    mu, sigma = nothing, nothing
    if mode == 0
        d = ndims(x) == 4 ? (1,2,4) : (2,)
        s = prod(size(x,d...))
        mu = sum(x,d) / s
        x0 = x .- mu
        x1 = x0 .* x0
        sigma = sqrt(epsilon + (sum(x1, d)) / s)
    elseif mode == 1
        mu = popfirst!(ms)
        sigma = popfirst!(ms)
    end

    # we need value in backpropagation
    push!(ms, value(mu), value(sigma))
    xhat = (x.-mu) ./ sigma
    return w[1] .* xhat .+ w[2]
end

function reslayerx0(w,x,ms; padding=0, stride=1, mode=1)
    b  = conv4(w[1],x; padding=padding, stride=stride)
    bx = batchnorm(w[2:3],b,ms; mode=mode)
end

function reslayerx1(w,x,ms; padding=0, stride=1, mode=1)
    relu.(reslayerx0(w,x,ms; padding=padding, stride=stride, mode=mode))
end

function reslayerx2(w,x,ms; pads=[0,1,0], strides=[1,1,1], mode=1)
    ba = reslayerx1(w[1:3],x,ms; padding=pads[1], stride=strides[1], mode=mode)
    bb = reslayerx1(w[4:6],ba,ms; padding=pads[2], stride=strides[2], mode=mode)
    bc = reslayerx0(w[7:9],bb,ms; padding=pads[3], stride=strides[3], mode=mode)
end

function reslayerx3(w,x,ms; pads=[0,0,1,0], strides=[2,2,1,1], mode=1) # 12
    a = reslayerx0(w[1:3],x,ms; stride=strides[1], padding=pads[1], mode=mode)
    b = reslayerx2(w[4:12],x,ms; strides=strides[2:4], pads=pads[2:4], mode=mode)
    relu.(a .+ b)
end

function reslayerx4(w,x,ms; pads=[0,1,0], strides=[1,1,1], mode=1)
    relu.(x .+ reslayerx2(w,x,ms; pads=pads, strides=strides, mode=mode))
end

function reslayerx5(w,x,ms; strides=[2,2,1,1], mode=1)
    x = reslayerx3(w[1:12],x,ms; strides=strides, mode=mode)
    for k = 13:9:length(w)
        x = reslayerx4(w[k:k+8],x,ms; mode=mode)
    end
    return x
end

function get_params(params, atype)
    len = length(params["value"])
    ws, ms = [], []
    for k = 1:len
        name = params["name"][k]
        value = convert(Array{Float32}, params["value"][k])

        if endswith(name, "moments")
            push!(ms, reshape(value[:,1], (1,1,size(value,1),1)))
            push!(ms, reshape(value[:,2], (1,1,size(value,1),1)))
        elseif startswith(name, "bn")
            push!(ws, reshape(value, (1,1,length(value),1)))
        elseif startswith(name, "fc") && endswith(name, "filter")
            push!(ws, transpose(reshape(value,(size(value,3),size(value,4)))))
        elseif startswith(name, "conv") && endswith(name, "bias")
            push!(ws, reshape(value, (1,1,length(value),1)))
        else
            push!(ws, value)
        end
    end
    map(wi->convert(atype, wi), ws),
    map(mi->convert(atype, mi), ms)
end
