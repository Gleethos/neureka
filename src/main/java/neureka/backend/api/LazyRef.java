package neureka.backend.api;

import java.util.function.Supplier;

/**
 * 	This will simply fetch a variable from a lambda once and then continuously
 * 	return this one value.
 * 	In a sense it is a lazy reference!
 * 	This is an internal class, do not depend on this outside this package.
 *
 * @param <V> The value type parameter of the thing wrapped by this.
 */
public final class LazyRef<V>
{
    public static <V> LazyRef<V> of( Supplier<V> source ) { return new LazyRef<>(source); }

    private Supplier<V> _source;
    private V _variable = null;

    private LazyRef(Supplier<V> source) { _source = source; }

    public V get() {
        if ( _source == null ) return _variable;
        else {
            _variable = _source.get();
            _source = null;
        }
        return _variable;
    }

    @Override
    public String toString() {
        String prefix = getClass().getSimpleName();
        if ( _variable == null ) return prefix + "<>[?]";
        try {
            V value = this.get();
            return prefix + "<" + value.getClass().getSimpleName() + ">" + "[" + value + "]";
        } catch (Exception e) {
            return prefix + "<>[?]";
        }
    }

}
