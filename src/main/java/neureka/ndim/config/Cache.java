package neureka.ndim.config;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.Function;

/**
 *  This is a simple internal cache for many objects which are
 *  mostly the same throughout the library runtime...
 *
 * @param <O> The type that should be cached, this may be an {@link neureka.ndim.config.NDConfiguration} or {@code int[]} array.
 */
public class Cache<O> {

    private final Object[] _ringBuffer;
    private int _size = 0;

    public Cache( int size ) {
        _ringBuffer = new Object[ size ];
    }

    public O process( O newObject ) {

        int index = _indexFor(newObject);

        O found = _getAt(index);

        if ( _equalsFor( found, newObject ) ) return found;
        else _setAt( index, newObject );

        return newObject;
    }

    public boolean has( O o ) {
        O found = (O) _ringBuffer[ _indexFor( o ) ];
        return found != null && found.equals( o );
    }

    public int size() { return _size; }

    private O _getAt( int index ) {
        return (O) _ringBuffer[ index ];
    }

    private void _setAt( int index, O o ) {
        if ( _ringBuffer[ index ] == null && o != null ) _size++;
        _ringBuffer[ index ] = o;
    }

    private boolean _equalsFor( Object a, Object b ) {
        if ( a != null && b != null ) {
            if (a instanceof int[] && b instanceof int[])
                return Arrays.equals((int[]) a, (int[]) b);
            else
                return Objects.equals(a, b);
        }
        else return false;
    }

    private int _indexFor( Object o ) {
        return o instanceof int[] ? _index((int[]) o) : _index( o.hashCode() );
    }

    private int _index( int key ) {
        return Math.abs(key)%_ringBuffer.length;
    }

    private int _index( int[] data )
    {
        long key = 0;
        for ( int e : data ) {
            if      ( e <=              10 ) key *=              10;
            else if ( e <=             100 ) key *=             100;
            else if ( e <=           1_000 ) key *=           1_000;
            else if ( e <=          10_000 ) key *=          10_000;
            else if ( e <=         100_000 ) key *=         100_000;
            else if ( e <=       1_000_000 ) key *=       1_000_000;
            else if ( e <=      10_000_000 ) key *=      10_000_000;
            else if ( e <=     100_000_000 ) key *=     100_000_000;
            else if ( e <=   1_000_000_000 ) key *=   1_000_000_000;
            key += Math.abs( e ) + 1;
        }
        int rank = data.length;
        while ( rank != 0 ) {
            rank /= 10;
            key *= 10;
        }
        key += data.length;
        return _index(Long.valueOf(key).hashCode());
    }

    public static class LazyEntry<K,V> {

        private final K _key;
        private final Function<K,V> _valueSupplier;

        private V _value = null;

        public LazyEntry( K directory, Function<K,V> valueSupplier ) {
            _key = directory;
            _valueSupplier = valueSupplier;
        }

        public V getValue() {
            if ( _value == null ) _value = _valueSupplier.apply(_key);
            return _value;
        }

        @Override
        public boolean equals(Object o) {
            if ( this == o ) return true;
            if ( o == null || getClass() != o.getClass() ) return false;
            LazyEntry that = (LazyEntry) o;
            return _key.equals(that._key);
        }

        @Override
        public int hashCode() {
            return Objects.hash(_key);
        }

    }

}
