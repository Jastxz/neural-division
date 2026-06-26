"""
Abstracción de backend GPU para el Método de la División Neuronal.

Soporta tres backends:
  - :cpu    → Arrays estándar de Julia (default)
  - :cuda   → CUDA.jl para GPUs NVIDIA
  - :rocm   → AMDGPU.jl para GPUs AMD (ROCm)

El backend se selecciona automáticamente según el hardware disponible,
o se puede forzar manualmente con `seleccionar_backend!(:cuda)`.

Uso:
    # Automático (detecta GPU disponible)
    inicializar_gpu!()

    # Manual
    seleccionar_backend!(:rocm)

    # Mover datos a GPU
    datos_gpu = a_dispositivo(datos_cpu)

    # Mover datos de vuelta a CPU
    datos_cpu = a_cpu(datos_gpu)
"""

# Backend activo (global mutable)
const _BACKEND = Ref{Symbol}(:cpu)

"""
    backend_actual() -> Symbol

Devuelve el backend activo: `:cpu`, `:cuda`, o `:rocm`.
"""
backend_actual() = _BACKEND[]

"""
    seleccionar_backend!(backend::Symbol)

Fuerza un backend específico. Opciones: `:cpu`, `:cuda`, `:rocm`.
"""
function seleccionar_backend!(backend::Symbol)
    backend in (:cpu, :cuda, :rocm) || error("Backend no soportado: $backend. Usar :cpu, :cuda, o :rocm")
    if backend == :cuda
        _verificar_cuda()
    elseif backend == :rocm
        _verificar_rocm()
    end
    _BACKEND[] = backend
    @info "Backend GPU seleccionado: $backend"
    return backend
end

"""
    inicializar_gpu!() -> Symbol

Detecta automáticamente el hardware GPU disponible y selecciona el mejor backend.
Prioridad: CUDA > ROCm > CPU.
"""
function inicializar_gpu!()
    # Intentar CUDA primero
    if _cuda_disponible()
        _BACKEND[] = :cuda
        @info "GPU NVIDIA detectada, usando backend CUDA"
        return :cuda
    end

    # Intentar ROCm
    if _rocm_disponible()
        _BACKEND[] = :rocm
        @info "GPU AMD detectada, usando backend ROCm"
        return :rocm
    end

    # Fallback a CPU
    _BACKEND[] = :cpu
    @info "No se detectó GPU, usando CPU"
    return :cpu
end

# ============================================================================
# Detección de hardware
# ============================================================================

function _cuda_disponible()
    try
        @eval using CUDA
        return CUDA.functional()
    catch
        return false
    end
end

function _rocm_disponible()
    try
        @eval using AMDGPU
        return AMDGPU.functional()
    catch
        return false
    end
end

function _verificar_cuda()
    _cuda_disponible() || error("CUDA no disponible. Instalar: using Pkg; Pkg.add(\"CUDA\")")
end

function _verificar_rocm()
    _rocm_disponible() || error("AMDGPU/ROCm no disponible. Instalar: using Pkg; Pkg.add(\"AMDGPU\")")
end

# ============================================================================
# Transferencia de datos CPU ↔ GPU
# ============================================================================

"""
    a_dispositivo(x::AbstractArray) -> AbstractArray

Mueve un array al dispositivo activo (GPU o CPU).
"""
function a_dispositivo(x::AbstractArray)
    b = _BACKEND[]
    b == :cpu && return x
    if b == :cuda
        @eval using CUDA
        return CUDA.CuArray(x)
    elseif b == :rocm
        @eval using AMDGPU
        return AMDGPU.ROCArray(x)
    end
    return x
end

"""
    a_dispositivo(datos::NamedTuple) -> NamedTuple

Mueve un NamedTuple de datos (entradas, salidas) al dispositivo activo.
"""
function a_dispositivo(datos::NamedTuple)
    return (entradas=a_dispositivo(datos.entradas), salidas=a_dispositivo(datos.salidas))
end

"""
    a_cpu(x::AbstractArray) -> Array

Mueve un array de vuelta a CPU.
"""
function a_cpu(x::AbstractArray)
    b = _BACKEND[]
    b == :cpu && return x
    return Array(x)
end

"""
    a_cpu(datos::NamedTuple) -> NamedTuple

Mueve un NamedTuple de datos de vuelta a CPU.
"""
function a_cpu(datos::NamedTuple)
    return (entradas=a_cpu(datos.entradas), salidas=a_cpu(datos.salidas))
end

# ============================================================================
# Operaciones matriciales agnósticas al backend
# ============================================================================

"""
    crear_zeros(T, dims...) -> AbstractArray

Crea un array de zeros en el dispositivo activo.
"""
function crear_zeros(::Type{T}, dims...) where T
    x = zeros(T, dims...)
    return a_dispositivo(x)
end

"""
    crear_randn(T, dims...; seed=nothing) -> AbstractArray

Crea un array de valores aleatorios normales en el dispositivo activo.
"""
function crear_randn(::Type{T}, dims...; seed=nothing) where T
    seed !== nothing && Random.seed!(seed)
    x = randn(T, dims...)
    return a_dispositivo(x)
end

"""
    info_dispositivo() -> String

Devuelve información sobre el dispositivo activo.
"""
function info_dispositivo()
    b = _BACKEND[]
    if b == :cpu
        return "CPU ($(Sys.CPU_NAME !== nothing ? Sys.CPU_NAME : "desconocido"))"
    elseif b == :cuda
        try
            @eval using CUDA
            dev = CUDA.device()
            return "CUDA: $(CUDA.name(dev)) ($(round(CUDA.totalmem(dev) / 1e9, digits=1)) GB)"
        catch
            return "CUDA (info no disponible)"
        end
    elseif b == :rocm
        try
            @eval using AMDGPU
            dev = AMDGPU.device()
            return "ROCm: $(AMDGPU.device_name(dev))"
        catch
            return "ROCm (info no disponible)"
        end
    end
    return "Desconocido"
end
