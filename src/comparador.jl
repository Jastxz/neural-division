"""
    es_mejor(nueva::Subconfiguracion, nueva_prec::T, actual::EntradaMapaSoluciones{T})::Bool where T

Determina si una nueva subconfiguración es mejor que la almacenada actualmente en una entrada
del mapa de soluciones, según el criterio de simplicidad:

1. Si la entrada actual no tiene subconfiguración (`nothing`), la nueva es mejor.
2. Si la nueva tiene menos neuronas activas, es mejor.
3. Si tienen las mismas neuronas activas y la nueva tiene mayor precisión, es mejor.
4. En cualquier otro caso, no es mejor.
"""
function es_mejor(nueva::Subconfiguracion, nueva_prec::T, actual::EntradaMapaSoluciones{T})::Bool where T
    actual.subconfiguracion === nothing && return true
    nueva.n_neuronas_activas < actual.subconfiguracion.n_neuronas_activas && return true
    nueva.n_neuronas_activas == actual.subconfiguracion.n_neuronas_activas && return nueva_prec > actual.precision
    return false
end
