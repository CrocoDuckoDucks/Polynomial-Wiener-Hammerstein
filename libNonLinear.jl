using FFTW
using DSP
using Plots
using CSV

function lin2dB(value::Number, reference::Number = 1.0)
    return 20.0 * log10(abs(value / reference))
end

function dB2lin(value::Number, reference::Number = 1.0)
    return abs(reference) * exp10(value / 20.0)
end

function wrapTo2Pi!(x::AbstractArray)
    x .-= 2pi .* fld.(x, 2pi)
end

function wrapTo2Pi(x::AbstractArray)
    return wrapTo2Pi!(copy(x))
end

function fcircshiftExp(x::AbstractArray, s::Real)

    N = length(x)
    nyq = div(N, 2) + 1

    ph = collect(2pi .* (0:(nyq - 1)) .* s ./ N)
    wrapTo2Pi!(ph)

    phFactor = ones(Complex, N)
    phFactor[1:nyq] = exp.(-im .* ph)
    phFactor[mod.(N .+ 1 .- (1:nyq), N) .+ 1]  = conj.(phFactor[1:nyq])

    return phFactor

end

function fcircshift(x::AbstractArray, s::Real)
    return ifft(fft(x) .* fcircshiftExp(x, s))
end

function fcircshiftExp(x::AbstractArray{<:Real}, s::Real)

    N = length(x)
    nyq = div(N, 2) + 1

    ph = collect(2pi .* (0:(nyq - 1)) .* s ./ N)
    wrapTo2Pi!(ph)

    phFactor = ones(Complex, nyq)
    phFactor = exp.(-im .* ph)

    return phFactor

end

function fcircshift(x::AbstractArray{<:Real}, s::Real)
    return irfft(rfft(x) .* fcircshiftExp(x, s), N)
end

function cheb(order::Integer, x::AbstractArray{<:Real})

    x = float(x)

    if isempty(x)
        return empty(x)
    end

    T = zeros(size(x))

    T[abs.(x) .<= one(eltype(x))] = cospi.(
        order * acos.(x[abs.(x) .<= one(eltype(x))]) / pi
        )

    T[x .> one(eltype(x))] = cosh.(
        order * acosh.(x[x .> one(eltype(x))])
        )

    T[x .< -one(eltype(x))] = (1 - 2mod(order, 2)) * cosh.(
        order * acosh.(-x[x .< -one(eltype(x))])
        )

    return T

end

function chebwin(M::Integer, at::Real = 100.0)

    if M < 1
        return ones(0)
    end

    if M == 1
        return ones(1)
    end

    odd = mod(M, 2) != 0

    order = M - 1
    beta = cosh(acosh(exp10(at / 20.0)) / order)

    k = 0:(M - 1)
    x = beta .* cospi.(k ./ M)

    p = cheb(order, x) ./ cosh(order * acosh(beta))

    if odd
        w = real(fft(p))
        n = div(M + 1, 2)
        w = [w[n:-1:2]; w[1:n]] ./ w[1]
    else
        w = real(fft(p .* exp.(im * pi * k / M)))
        n = div(M, 2) + 1
        w = [w[n:-1:2]; w[2:n]] ./ w[2]
    end

    return w

end

function branchAntialFirTaps(
    nBranch::Integer,
    nTaps::Integer,
    ripple::Real
    )

    rType = Lowpass((nTaps - 1) / (nBranch * nTaps))
    dMeth = FIRWindow(hanning(nTaps)) #FIRWindow(chebwin(nTaps, ripple))

    return digitalfilter(rType, dMeth)

end

function randomUniform(a::Real, b::Real)
    return (b - a) * rand() + a
end

function randomFrequency(fMin::Real = 1e-4)

    a = log10(fMin)
    b = log10(1.0 - eps(1.0))

    return exp10(randomUniform(a, b))

end

function randomdBGain(minGain_dB::Real, maxGain_dB::Real)
    return randomUniform(minGain_dB, maxGain_dB)
end

function randomLinGain(minGain_dB::Real, maxGain_dB::Real)
    return dB2lin(randomdBGain(minGain_dB, maxGain_dB))
end

function randomLowpass(fMin::Real)
    return Lowpass(randomFrequency(fMin))
end

function randomHighpass(fMin::Real)
    return Highpass(randomFrequency(fMin))
end

function randomBandpass(fMin::Real)

    corners = sort([randomFrequency(fMin); randomFrequency(fMin)])

    return Bandpass(corners[1], corners[2])

end

function randomBandstop(fMin::Real)

    corners = sort([randomFrequency(fMin); randomFrequency(fMin)])

    return Bandstop(corners[1], corners[2])

end

function randomResponseType(fMin::Real)

    dTypes = [
        randomLowpass;
        randomHighpass;
        randomBandpass;
        randomBandstop
        ]

    return dTypes[rand(1:end)](fMin)

end

function randomInteger(maxInt::Integer)
    return rand(1:maxInt)
end

function randomOrder(maxOrder::Integer)
    return randomInteger(maxOrder)
end

function randomRipple(rippleLim1::Real, rippleLim2::Real)
    return randomUniform(rippleLim1, rippleLim2)
end

function randomButterworth(maxOrder::Integer)
    return Butterworth(randomOrder(maxOrder))
end

function randomChebyshev1(maxOrder::Integer, rippleLim1::Real, rippleLim2::Real)

    return Chebyshev1(
        randomOrder(maxOrder),
        randomRipple(rippleLim1, rippleLim2)
        )

end

function randomChebyshev2(maxOrder::Integer, rippleLim1::Real, rippleLim2::Real)

    return Chebyshev2(
        randomOrder(maxOrder),
        randomRipple(rippleLim1, rippleLim2)
        )

end

function randomElliptic(maxOrder::Integer, rippleLim1::Real, rippleLim2::Real)

    ripples = sort(
    [randomRipple(rippleLim1, rippleLim2);
    randomRipple(rippleLim1, rippleLim2)]
    )

    return Elliptic(randomOrder(maxOrder), ripples[1], ripples[2])

end

function randomDesignMethod(
    maxOrder::Integer,
    rippleLim1::Real,
    rippleLim2::Real
    )

    coin = rand()

    if coin <= 0.5

        return randomButterworth(maxOrder)

    else

        dMethods = [
            randomChebyshev1;
            randomChebyshev2;
            randomElliptic
            ]

            return dMethods[rand(1:end)](maxOrder, rippleLim1, rippleLim2)

    end

end

function randomIIR(
    fMin::Real,
    maxOrder::Integer,
    rippleLim1::Real,
    rippleLim2::Real
    )

    return digitalfilter(
        randomResponseType(fMin),
        randomDesignMethod(maxOrder, rippleLim1, rippleLim2)
        )

end

struct pwh
    N::Integer
    antiAls
    kernels
    gains
end

function randomPWH(
    N::Integer,
    nTaps::Integer,
    ripple::Real,
    fMin::Real,
    maxOrder::Integer,
    rippleLim1::Real,
    rippleLim2::Real,
    minGain_dB::Real,
    maxGain_dB::Real
    )

    antiAls = []
    kernels = []
    gains   = []

    for n in 1:N

        append!(antiAls, [branchAntialFirTaps(n, nTaps, ripple)])
        append!(kernels, [randomIIR(fMin, maxOrder, rippleLim1, rippleLim2)])
        append!(gains, randomLinGain(minGain_dB, maxGain_dB))

    end

    return pwh(N, antiAls, kernels, gains)

end

function writeSos(
    fID::IOStream,
    sos::SecondOrderSections,
    biqFcn::String = "tf2np"
    )

    write(fID, "*(_, $(sos.g)) : ")

    for b = 1:length(sos.biquads)

        write(fID, "fi.$biqFcn(")

        write(fID, "$(sos.biquads[b].b0), $(sos.biquads[b].b1), $(sos.biquads[b].b2), $(sos.biquads[b].a1), $(sos.biquads[b].a2))")

        if b != length(sos.biquads)
            write(fID, " : ")
        end

    end

end

function pwh2faust(pwhObj::pwh, dspPath::String, biqFcn::String)

    open(dspPath, "w") do fID

        write(fID, """fi = library("filters.lib");\nde = library("delays.lib");\n\n""")

        for n in 1:pwhObj.N

            sos = convert(SecondOrderSections, pwhObj.kernels[n])

            write(fID, "br$n = fi.fir((")

            for t in 1:length(pwhObj.antiAls[n])

                write(fID, "$(pwhObj.antiAls[n][t])")

                if t == length(pwhObj.antiAls[n])
                    write(fID, ")) : pow(_, $(n)) : *(_, $(pwhObj.gains[n])) : ")
                else
                    write(fID, ", ")
                end

            end

            writeSos(fID, sos, biqFcn)

            write(fID, ";\n")

        end

        write(fID, "\nprocess = _ <: ")

        for n in 1:pwhObj.N

            write(fID, "br$(n)")

            if n == length(pwhObj.antiAls)
                write(fID, " :> _;\n")
            else
                write(fID, ", ")
            end

        end

    end

end

function pwh2plot(pwhObj::pwh, wLength::Integer, Fs::Real)

    plt = plot(layout = 4)

    plot!(
        plt,
        ylabel = "Magnitude [dB]",
        title = "Branch Antialiasing",
        xlabel = "Frequency [half-cycles/sample]",
        subplot = 1
        )

    plot!(
        plt,
        ylabel = "Angle [rad]",
        title = "Branch Antialiasing",
        xlabel = "Frequency [half-cycles/sample]",
        subplot = 3
        )

    plot!(
        plt,
        ylabel = "Magnitude [dB]",
        title = "Branch Kernel",
        xlabel = "Frequency [Hz]",
        subplot = 2
        )

    plot!(
        plt,
        ylabel = "Angle [rad]",
        title = "Branch Kernel",
        xlabel = "Frequency [Hz]",
        subplot = 4
        )

    fKernels = (0:(wLength - 1)) * Fs / 2wLength

    for n in 1:pwhObj.N

        H = rfft(pwhObj.antiAls[n])

        fAntial = (0:(length(H) - 1)) / length(H)

        plot!(
            plt,
            fAntial[2:end],
            lin2dB.(H[2:end]),
            label = "$n",
            subplot = 1
            )

        plot!(
            plt,
            fAntial[2:end],
            unwrap(angle.(H[2:end])),
            label = "$n",
            subplot = 3
            )

        K = pwhObj.gains[n] .* freqz(pwhObj.kernels[n], fKernels, Fs)

        plot!(
            plt,
            fKernels[2:end],
            lin2dB.(K[2:end]),
            xscale = :log10,
            label = "$n",
            subplot = 2
            )

        plot!(
            plt,
            fKernels[2:end],
            unwrap(angle.(K[2:end])),
            xscale = :log10,
            label = "$n",
            subplot = 4
            )

    end

    return plt

end

function pwh2matrix(pwhObj::pwh, wLength::Integer, Fs::Real)

    G           = zeros(Complex{Float64}, length(pwhObj.antiAls), wLength)
    fKernels    = (0:(wLength - 1)) * Fs / 2wLength

    for n in 1:length(pwhObj.antiAls)
        G[n, :] = pwhObj.gains[n] .* freqz(pwhObj.kernels[n], fKernels, Fs)
    end

    return G, fKernels

end

function readMatrixCsv(rePath::String, imPath::String)

    Mre = convert(Array{Complex{Float64}, 2}, CSV.read("$rePath", header = 0))
    Mim = im * convert(Array{Float64, 2}, CSV.read("$imPath", header = 0))

    return  Mre + Mim

end

function matrix2plot(M::AbstractArray, Fs::Real)

    nyq = div(size(M, 2), 2) + 1

    f = (0:(nyq - 1)) * Fs / 2nyq

    plt = plot(layout = 2, xlabel = "Frequency [Hz]")

    plot!(
        plt,
        f[2:end],
        transpose(lin2dB.(M[:, 2:nyq])),
        ylabel = "Magnitude [dB]",
        xscale = :log10,
        subplot = 1
        )

    plot!(
        plt,
        f[2:end],
        unwrap(transpose(angle.(M[:, 2:nyq])), dims = 1),
        ylabel = "Angle [rad]",
        xscale = :log10,
        subplot = 2
        )

    return plt

end

function cMatrix(N::Integer, A::Real = 1.0)

    C = zeros(Complex, N, N)

    for r in 1:N

        a = A^(r - 1)

        for c in r:N

            if !iseven(r + c)
                continue
            end

            C[r, c] =
                a *
                Complex(-1)^(2c + (1 - r) / 2) *
                exp2(1 - c) *
                binomial(c, div(c - r, 2))

        end

    end

    return C

end

function novakWin(M::Integer, I::Integer, O::Integer)

    @assert I <= (M - O + 1)

    w = ones(M)

    w[1:(I - 1)] =
        0.5 * sinpi.((2((1:(I - 1)) .- 1) .- I .+ 1) ./ (2(I - 1))) .+ 0.5


    w[end:-1:(end - O + 2)] =
        0.5 * sinpi.((2((1:(O - 1)) .- 1) .- O .+ 1) ./ (2(O - 1))) .+ 0.5

    return w

end

function higherGap(γ::Real, n::Integer, Fs::Real)
    return Fs * γ * log(n)
end

function gap2next(γ::Real, n::Integer, Fs::Real)
    return Fs * γ * log(1.0 + 1.0 / n)
end

function gap2previous(γ::Real, n::Integer, Fs::Real)
    Fs * γ * log(n / (n - 1))
end

function windowHighers(
    h::AbstractArray,
    M::Integer,
    N::Integer,
    γ::Real,
    Fs::Real,
    innerWin::Function,
    outerWin::Function
    )

    c = div(length(h), 2)

    H = zeros(Complex{Float64}, N, M)

    w = zeros(M)

    for n in 1:N

        fill!(w, zero(eltype(w)))

        fCenter         = c - higherGap(γ, n, Fs)
        fGap2Next       = gap2next(γ, n, Fs)
        fGap2Previous   = gap2previous(γ, n, Fs)

        fOptHead        = fCenter - min(0.5 * fGap2Next, 0.5 * M)
        fOptTail        = fCenter + min(0.5 * fGap2Previous, 0.5 * M)
        fOptLength      = fOptTail - fOptHead
        nMaxData        = floor(Integer, fOptLength)

        nOptHead        = floor(Integer, fOptHead)
        nOptTail        = floor(Integer, fOptTail)
        fOptCount       = floor(Integer, fOptLength)

        fHead           = fOptHead
        nHead           = nOptHead

        fWhead          = 0.5 * (M - min(fGap2Next, M)) + 1
        nWhead          = floor(Integer, fWhead)

        cShift          = nHead - fHead
        wShift          = fWhead - nWhead

        alignShift      = cShift + wShift
        totalShift      = alignShift + 0.5 * M

        w[nWhead:(nWhead + nMaxData - 1)] = h[nHead:(nHead + nMaxData - 1)] .*
            innerWin(nMaxData)

        H[n, :] = fft(w .* outerWin(M))
        H[n, :] .*= fcircshiftExp(H[n, :], totalShift)

    end

    return H

end

function hammSolve(
    h::AbstractArray,
    M::Integer,
    N::Integer,
    γ::Real,
    A::Real,
    Fs::Real,
    innerWin::Function,
    outerWin::Function
    )

    G = cMatrix(N, A) \ windowHighers(h, M, N, γ, Fs, innerWin, outerWin)

    # Force Hermitian Symmetry
    M = size(G, 2)
    nyq = div(M, 2) + 1
    G[:, mod.(M .+ 1 .- (1:nyq), M) .+ 1] = conj.(G[:, 1:nyq])

    return G

end

function coreMatrix(N::Integer, K::Integer)
    return 2.0 * (0:(N - 1)) * transpose(0:(K - 1)) / K
end

function firFreqz(
    b::AbstractVector{<:Real},
    K::Integer
    )

    return transpose(exp.(-im * π * coreMatrix(length(b), K))) * b

end

function costFunction(
    b::AbstractVector{<:Real},
    Hₜ::AbstractVector{<:Complex},
    C::AbstractMatrix{<:Real},
    S::AbstractMatrix{<:Real}
    )

    H = firFreqz(b, length(Hₜ))

    J   = sum(abs2, firFreqz(b, length(Hₜ)) - Hₜ)
    ∇J  = 2.0 * sum(
        C .* transpose(real(H - Hₜ)) - S .* transpose(imag(H - Hₜ)),
        dims=2
        )[:, 1]

    return J, ∇J

end

function firGradientDescent(
    Hₜ::AbstractVector{<:Complex},
    N::Integer,
    α::Function,
    I::Integer,
    ϵ::Real
    )

    b = zeros(N)
    b[1] = 1.0

    J = zeros(I)

    M = coreMatrix(N, length(Hₜ))

    C = cospi.(M)
    S = sinpi.(M)

    for i in 1:I

        J[i], ∇J = costFunction(b, Hₜ, C, S)

        b = b - α(i) * ∇J

        if (i > 1) && (abs(J[i - 1] - J[i]) < ϵ)
            break
        end

    end

    return b, J

end

function test_firGradientDescent()

    responsetype    = Lowpass(0.2)
    designmethod    = FIRWindow(hanning(64))
    testFilt        = digitalfilter(responsetype, designmethod)
    K               = 1024
    N               = length(testFilt)
    α(i)            = 1e-6
    I               = 3000

    Hₜ = firFreqz(testFilt, K)

    b, J = firGradientDescent(Hₜ, N, α, I)

    return b, J, testFilt

end

function test_firGradientDescent_2()

    responsetype    = Lowpass(0.2)
    designmethod    = Elliptic(4, 0.5, 30)
    testFilt        = digitalfilter(responsetype, designmethod)
    K               = 1024
    N               = 128
    α(i)            = 1e-6
    I               = 3000

    Hₜ = freqz(testFilt, 2π * (0:(K - 1)) / K)

    b, J = firGradientDescent(Hₜ, N, α, I)

    return b, J, testFilt

end

function hamm2Firs(
    G::AbstractMatrix{<:Complex},
    B::Integer,
    α::Function,
    I::Integer,
    ϵ::Real
    )

    bMat = zeros(B, size(G, 1))
    jMat = zeros(I, size(G, 1))

    for n in 1:size(G, 1)
        bMat[:, n], jMat[:, n] = firGradientDescent(G[n, :], B, α, I, ϵ)
    end

    return bMat, jMat

end

function hammIdentify(
    h::AbstractArray,
    M::Integer,
    N::Integer,
    γ::Real,
    A::Real,
    Fs::Real,
    innerWin::Function,
    outerWin::Function,
    B::Integer,
    α::Function,
    I::Integer,
    ϵ::Real
    )

    G = hammSolve(h, M, N, γ, A, Fs, innerWin, outerWin)

    return hamm2Firs(G, B, α, I, ϵ)

end
