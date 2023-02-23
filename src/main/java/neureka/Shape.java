package neureka;

import java.util.List;
import java.util.stream.Stream;

/**
 *  Basically a tuple of integers which is used to describe the shape of an array.
 *  The shape of an array is the number of elements in each dimension.
 *  A {@link Shape} is an immutable monadic type, which means that you can transform
 *  a {@link Shape} into another {@link Shape} by applying a function to it, like for example
 *  through the {@link #map(java.util.function.Function)} method.
 */
public interface Shape extends Iterable<Integer>
{
    /**
     *  This method is used to create a {@link Shape} instance from an array of integers.
     *  The array of integers is used to describe the shape of an array.
     *  The shape of an array is the number of elements in each dimension.
     *  @param shape The array of integers which is used to describe the shape of an array.
     *  @return A {@link Shape} instance which is created from the given array of integers.
     */
    static Shape of( int... shape ) {
        int[] data = shape.clone();
        return new Shape() {
            @Override public int size() { return data.length; }
            @Override public int get( int i ) { return data[i]; }
            @Override public String toString() { return java.util.Arrays.toString( data ); }
            @Override public boolean equals( Object o ) {
                if ( o instanceof Shape ) {
                    Shape s = (Shape) o;
                    if ( s.size() == size() ) {
                        for ( int i = 0; i < size(); i++ )
                            if ( s.get(i) != get(i) ) return false;
                        return true;
                    }
                }
                if ( o instanceof List) { // We also want to be able to compare to a list of integers!
                    List<Integer> l = (List<Integer>) o;
                    if ( l.size() == size() ) {
                        for ( int i = 0; i < size(); i++ )
                            if ( l.get(i) != get(i) ) return false;
                        return true;
                    }
                }
                return false;
            }
            @Override public int hashCode() {
                int hash = 0;
                for ( int i = 0; i < size(); i++ ) hash += get(i);
                return hash;
            }
        };
    }

    /**
     * @return The number of dimensions of the shape.
     */
    int size();

    /**
     * @param i The index of the dimension/axis.
     * @return The number of elements in the dimension/axis at the given index.
     */
    int get( int i );

    /**
     * @return This shape as an array of integers.
     */
    default int[] toIntArray() {
        int[] arr = new int[ size() ];
        for ( int i = 0; i < size(); i++ ) arr[i] = get(i);
        return arr;
    }

    /**
     *  This method is used to transform a {@link Shape} into another {@link Shape}
     *  by applying a function to it.
     *  @param mapper The function which is used to transform the {@link Shape}.
     *  @return A new {@link Shape} instance which is the result of the transformation.
     */
    default Shape map( java.util.function.Function<Integer, Integer> mapper ) {
        int[] arr = new int[ size() ];
        for ( int i = 0; i < size(); i++ ) arr[i] = mapper.apply( get(i) );
        return Shape.of( arr );
    }

    default Stream<Integer> stream() {
        return java.util.stream.IntStream.range(0, size()).mapToObj(this::get);
    }

    @Override
    default java.util.Iterator<Integer> iterator() {
        return new java.util.Iterator<Integer>() {
            int i = 0;
            @Override public boolean hasNext() { return i < size(); }
            @Override public Integer next() { return get(i++); }
        };
    }

}
