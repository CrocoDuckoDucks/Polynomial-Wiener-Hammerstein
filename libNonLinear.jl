using FFTW
using DSP
using Plots

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

function fcircshift(x::AbstractArray, s::Real)

    N = length(x)
    nyq = div(N, 2) + 1

    ph = collect(2pi .* (0:(nyq - 1)) .* s ./ N)
    wrapTo2Pi!(ph)

    phFactor = ones(Complex, N)
    phFactor[1:nyq] = exp.(-im .* ph)
    phFactor[mod.(N .+ 1 .- (1:nyq), N) .+ 1]  = conj.(phFactor[1:nyq])

    return ifft(fft(x) .* phFactor)

end

function fcircshift(x::AbstractArray{<:Real}, s::Real)

    N = length(x)
    nyq = div(N, 2) + 1

    ph = collect(2pi .* (0:(nyq - 1)) .* s ./ N)
    wrapTo2Pi!(ph)

    phFactor = ones(Complex, nyq)
    phFactor = exp.(-im .* ph)

    return irfft(rfft(x) .* phFactor, N)

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
    ripple::Real,
    Fs::Real
    )

    if nBranch == 1

        h = zeros(nTaps)
        h[div(nTaps, 2) + 1] = 1.0

        return h

    end

    rType = Lowpass(Fs / 2nBranch, fs = Fs)
    dMeth = FIRWindow(chebwin(nTaps, ripple))

    return digitalfilter(rType, dMeth)

end

function randomUniform(a::Real, b::Real)
    return (b - a) * rand() + a
end

function randomFrequency(Fs::Real, fMin::Real = 20.0)

    a = log10(fMin)
    b = log10(0.5 * Fs)

    return exp10(randomUniform(a, b))

end

function randomdBGain(minGain_dB::Real, maxGain_dB::Real)
    return randomUniform(minGain_dB, maxGain_dB)
end

function randomLinGain(minGain_dB::Real, maxGain_dB::Real)
    return dB2lin(randomdBGain(minGain_dB, maxGain_dB))
end

function randomLowpass(Fs::Real, fMin::Real)
    return Lowpass(randomFrequency(Fs, fMin), fs = Fs)
end

function randomHighpass(Fs::Real, fMin::Real)
    return Highpass(randomFrequency(Fs, fMin), fs = Fs)
end

function randomBandpass(Fs::Real, fMin::Real)

    corners = sort([randomFrequency(Fs, fMin); randomFrequency(Fs, fMin)])

    return Bandpass(corners[1], corners[2], fs = Fs)

end

function randomBandstop(Fs::Real, fMin::Real)

    corners = sort([randomFrequency(Fs, fMin); randomFrequency(Fs, fMin)])

    return Bandstop(corners[1], corners[2], fs = Fs)

end

function randomResponseType(Fs::Real, fMin::Real)

    dTypes = [
        randomLowpass;
        randomHighpass;
        randomBandpass;
        randomBandstop
        ]

    return dTypes[rand(1:end)](Fs, fMin)

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
    Fs::Real,
    fMin::Real,
    maxOrder::Integer,
    rippleLim1::Real,
    rippleLim2::Real
    )

    return digitalfilter(
        randomResponseType(Fs, fMin),
        randomDesignMethod(maxOrder, rippleLim1, rippleLim2)
        )

end

struct pwh
    antiAls
    kernels
    gains
end

function randomPWH(
    N::Integer,
    Fs::Real,
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

        append!(antiAls, [branchAntialFirTaps(n, nTaps, ripple, Fs)])
        append!(kernels, [randomIIR(Fs, fMin, maxOrder, rippleLim1, rippleLim2)])
        append!(gains, randomLinGain(minGain_dB, maxGain_dB))

    end

    return pwh(antiAls, kernels, gains)

end

function writeSos(fID::IOStream, sos::SecondOrderSections, biqFcn::String = "tf2np")

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

        for n in 1:length(pwhObj.antiAls)

            sos = convert(SecondOrderSections, pwhObj.kernels[n])

            if n == 1

                write(fID, "br1 = de.fdelay($(ceil(Integer, 0.5 * length(a.antiAls[1]))), $(0.5 * (length(a.antiAls[1]) - 1))) : ")

            else

                write(fID, "br$n = fi.fir((")

                for t in 1:length(pwhObj.antiAls[n])

                    write(fID, "$(pwhObj.antiAls[n][t])")

                    if t == length(pwhObj.antiAls[n])
                        write(fID, ")) : pow(_, $(n)) : *(_, $(pwhObj.gains[n])) : ")
                    else
                        write(fID, ", ")
                    end

                end

            end

            writeSos(fID, sos, biqFcn)

            write(fID, ";\n")

        end

        write(fID, "\nprocess = _ <: ")

        for n in 1:length(pwhObj.antiAls)

            write(fID, "br$(n)")

            if n == length(pwhObj.antiAls)
                write(fID, " :> fi.dcblockerat(10);\n")
            else
                write(fID, ", ")
            end

        end

    end

end

function pwh2plot(pwhObj::pwh, wLength::Integer, Fs::Real)

    plt = plot(layout = 4, xlabel = "Frequency [Hz]")

    plot!(
        plt,
        ylabel = "Magnitude [dB]",
        title = "Branch Antialiasing",
        subplot = 1
        )

    plot!(
        plt,
        ylabel = "Angle [rad]",
        title = "Branch Antialiasing",
        subplot = 3
        )

    plot!(
        plt,
        ylabel = "Magnitude [dB]",
        title = "Branch Kernel",
        subplot = 2
        )

    plot!(
        plt,
        ylabel = "Angle [rad]",
        title = "Branch Kernel",
        subplot = 4
        )

    fKernels = (0:(wLength - 1)) * Fs / 2wLength

    for n in 1:length(pwhObj.antiAls)

        H = rfft(pwhObj.antiAls[n])

        fAntial = (0:(length(H) - 1)) *
            Fs / 2length(H)

        plot!(
            plt,
            fAntial[2:end],
            lin2dB.(H[2:end]),
            xscale = :log10,
            label = "$n",
            subplot = 1
            )

        plot!(
            plt,
            fAntial[2:end],
            unwrap(angle.(H[2:end])),
            xscale = :log10,
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
