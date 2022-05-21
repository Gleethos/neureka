package neureka.backend.api;

import java.util.function.Supplier;

/**
 * 	This will simply fetch a variable from a lambda once and then continuously
 * 	return this one value.
 * 	In a sense it is a lazy reference!
 *
 * @param <V> The value type parameter of the thing wrapped by this.
 */
final class LazyRef<V> {

    public static <V> LazyRef<V> of(Supplier<V> source) { return new LazyRef<>(source); }

    private Supplier<V> _source;
    private V _variable = null;

    private LazyRef(Supplier<V> source) { this._source = source; }

    public V get() {
        if ( _source == null ) return _variable;
        else {
            _variable = _source.get();
            _source = null;
        }
        return _variable;
    }

    @Override
    public String toString() { return String.valueOf(this.get()); }

}
