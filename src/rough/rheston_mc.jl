using AQFED.MonteCarlo
import AQFED.Random: next!, nextn!, skipTo
using AQFED.Random
import SpecialFunctions: gamma
using Statistics


function logForward(model::RoughHestonParams, lnspot, t0)
  return lnspot
end

export simulateHQE

#simple version for now to compare with rough bergomi, later code vectorized version
function simulateHQE(
  rng,
  params::RoughHestonParams{T},
  spot::TS,
  payoff::MonteCarlo.VanillaOption,
  start::Int,
  nSim::Int,
  n::Int
) where {T,TS}
  specTimes = specificTimes(payoff)
  tte = specTimes[end]
  df = 1.0
  dt = 1.0 / n
  lnspot = log(spot)
  lnf0 = logForward(params, lnspot, tte)
  dimt = Int(floor(n * tte))
  genTimes = collect(range(0.0, stop=tte, length=(1 + dimt)))

  lam = zero(T)
  H = params.H
  eta = params.ν / (sqrt(2H) * gamma(H + 0.5))
  rho = params.ρ
  rho2m1 = sqrt(1 - rho * rho)
  eps0 = 1e-10
  W = zeros(dimt)
  Wperp = zeros(dimt)
  Z = zeros(dimt)
  U = zeros(dimt)
  Uperp = zeros(dimt)
  sqrtdt = sqrt(dt)
  tj = genTimes[2:end]
  xij = @. params.varianceCurve(tj)
  G0del = eta * G0(params, dt)
  # G1del = eta*G1(params,dt)
  G00del = eta^2 * G00(params, dt)
  # G11del = eta^2*G11(params,dt)
  # G01del = eta^2*G01(params,dt)
  G00j = eta^2 * vcat(0, G00.(Ref(params), tj))
  bstar = sqrt.(diff(G00j) / dt)
  # bstar1 = bstar[1] # bstar is average g over an interval
  rhovchi = G0del / sqrt(G00del * dt)
  betavchi = G0del / dt

  u = zeros(T,dimt)
  chi = zeros(T,dimt)
  #v <- rep(xi(0), paths)
  #xihat <- rep(xij[1], paths)
  v0 = params.varianceCurve(genTimes[1])
  payoffValues = Vector{Float64}(undef, nSim)
  specValues = Vector{Float64}(undef, length(specTimes))

  for sim = 1:nSim
    nextn!(rng, W)
    nextn!(rng, Wperp)
    nextn!(rng, Z)
    next!(rng, U)
    next!(rng, Uperp)

    x = zero(T)
    y = zero(T)
    w = zero(T)

    v = v0
    x = lnf0
    xihat = xij[1]
    specIndex = 1
    t0 = genTimes[1]
    for j = 1:dimt
      t1 = genTimes[j+1]

      xibar = (xihat + 2 * H * v) / (1 + 2 * H)
      eps = xibar * G00del * (1 - rhovchi^2)

      # Ben Wood bug fixes are in the two succeeding lines
      psichi = 4 * G00del * rhovchi^2 * xibar / xihat^2
      psieps = 4 * G00del * (1 - rhovchi^2) * xibar / xihat^2

      zchi = ifelse(psichi < 1.5, psiM(psichi, xihat / 2, W[j]), psiP(psichi, xihat / 2, U[j]))
      zeps = ifelse(psieps < 1.5, psiM(psieps, xihat / 2, Wperp[j]), psiP(psieps, xihat / 2, Uperp[j]))

      chi[j] = (zchi - xihat / 2) / betavchi
      eps = zeps - xihat / 2
      u[j] = betavchi * chi[j] + eps
      vf = xihat + u[j]
      if vf <= eps0
         vf = eps0
      end
      dw = (v + vf) / 2 * dt
      w = w + dw
      y = y + chi[j]
      x = x - dw / 2 + sqrt(dw) * rho2m1 * Z[j] + rho * chi[j]
      if j < dimt
        xihat = xij[j+1] + sum(bstar[j+1:-1:2] .* chi[1:j])        
      end
      v = vf

      if specIndex <= length(specTimes) && t1 >= specTimes[specIndex] - 1e-8
        specValues[specIndex] = exp(x)
        if specIndex == length(specTimes)
          payoffValues[sim] = evaluatePayoffOnPath(payoff, specValues, df)
        else
          specIndex += 1
        end
      end
      t0 = t1
    end
  end
  payoffMean = mean(payoffValues)
  return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))

end



function simulateHQEPath(
  rng,
  params::RoughHestonParams{T},
  spot::TS,
  tte,
  n::Int
) where {T,TS}
  dt = 1.0 / n

  lnspot = log(spot)
  lnf0 = logForward(params, lnspot, tte)
  dimt = Int(floor(n * tte))
  genTimes = collect(range(0.0, stop=tte, length=(1 + dimt)))

  H = params.H
  eta = params.ν / (sqrt(2H) * gamma(H + 0.5))
  rho = params.ρ
  rho2m1 = sqrt(1 - rho * rho)
  eps0 = 1e-10
  W = zeros(dimt)
  Wperp = zeros(dimt)
  Z = zeros(dimt)
  U = zeros(dimt)
  Uperp = zeros(dimt)
  sqrtdt = sqrt(dt)
  tj = genTimes[2:end]
  xij = @. params.varianceCurve(tj)
  G0del = eta * G0(params, dt)
  # G1del = eta*G1(params,dt)
  G00del = eta^2 * G00(params, dt)
  # G11del = eta^2*G11(params,dt)
  # G01del = eta^2*G01(params,dt)
  G00j = eta^2 * vcat(0, G00.(Ref(params), tj))
  bstar = sqrt.(diff(G00j) / dt)
  # bstar1 = bstar[1] # bstar is average g over an interval
  rhovchi = G0del / sqrt(G00del * dt)
  betavchi = G0del / dt

  u = zeros(T,dimt)
  chi = zeros(T,dimt)
  #v <- rep(xi(0), paths)
  #xihat <- rep(xij[1], paths)
  v0 = params.varianceCurve(genTimes[1])
  logpath = zeros(T,dimt+1)
  vpath = zeros(T,dimt+1)
  logpath[1] = lnf0
  vpath[1] = v0
    nextn!(rng, W)
    nextn!(rng, Wperp)
    nextn!(rng, Z)
    next!(rng, U)
    next!(rng, Uperp)

    x = zero(T)
    y = zero(T)
    w = zero(T)

    v = v0
    x = lnf0
    xihat = xij[1]
    t0 = genTimes[1]
    for j = 1:dimt
      t1 = genTimes[j+1]

      xibar = (xihat + 2 * H * v) / (1 + 2 * H)
      eps = xibar * G00del * (1 - rhovchi^2)

      # Ben Wood bug fixes are in the two succeeding lines
      psichi = 4 * G00del * rhovchi^2 * xibar / xihat^2
      psieps = 4 * G00del * (1 - rhovchi^2) * xibar / xihat^2

      zchi = ifelse(psichi < 1.5, psiM(psichi, xihat / 2, W[j]), psiP(psichi, xihat / 2, U[j]))
      zeps = ifelse(psieps < 1.5, psiM(psieps, xihat / 2, Wperp[j]), psiP(psieps, xihat / 2, Uperp[j]))

      chi[j] = (zchi - xihat / 2) / betavchi
      eps = zeps - xihat / 2
      u[j] = betavchi * chi[j] + eps
      vf = xihat + u[j]
      if vf <= eps0
         vf = eps0
      end
      dw = (v + vf) / 2 * dt
      w = w + dw
      y = y + chi[j]
      x = x - dw / 2 + sqrt(dw) * rho2m1 * Z[j] + rho * chi[j]
      if j < dimt
        xihat = xij[j+1] + sum(bstar[j+1:-1:2] .* chi[1:j])        
      end
      v = vf
      logpath[j+1] = x
      vpath[j+1] = v
      t0 = t1
    end
  
  return genTimes, logpath, vpath

end




function psiM(psi::T, ev::T, w::T) where {T}
  beta2 = 2 / psi - 1 + sqrt(2 / psi) * sqrt(abs(2 / psi - 1)) # The abs fixes situations where psi > 2
  alpha = ev / (1 + beta2)
  vf = alpha * (sqrt(abs(beta2)) + w)^2
  return vf
end

function psiP(psi::T, ev::T, u::T) where {T}
  p = 2 / (1 + psi)
   
  vf = if (u < p)
    gam = ev / 2 * (1 + psi)
    - gam * log(u / p)
  else
    zero(T)
  end
  return vf
end

function gGamma(params::RoughHestonParams{T}, tau::T) where {T}
  H = params.H
  al = H + 0.5
  return sqrt(2 * H) * tau^(al - 1)
end

function G00(params::RoughHestonParams{T}, tau::T) where {T}
  H = params.H
  return tau^(2 * H)
end

function G11(params::RoughHestonParams{T}, tau) where {T}
  H = params.H
  H2 = 2H
  return tau^(2H) * (2^H2 - 1)
end

# G0
function G0(params::RoughHestonParams{T}, dt) where {T}
  H = params.H
  al = H + 1 / 2
  return sqrt(2 * H) / al * dt^(al)
end

function G1(params::RoughHestonParams{T}, dt) where {T}
  H = params.H
  al = H + 1 / 2
  return sqrt(2H) / al * dt^(al) * (2^al - 1)
end

# Gamma covariance
function G0K(params::RoughHestonParams{T}, k, t; quad::Quadrature=GaussKronrod()) where {T}
  gp = x -> gGamma(params, x)
  integr = function (s)
    g(s) * gp(s + k * t)
  end
  res = integrate(quad, integr, zero(t), t)
  return res
end

# Gamma first order covariance
function G01(params::RoughHestonParams{T}, t; quad::Quadrature=GaussKronrod()) where {T}
  gp = x -> gGamma(params, x)
  integr = function (s)
    gp(s) * gp(s + t)
  end
  res = integrate(quad, integr, zero(t), t)
  return res
end
