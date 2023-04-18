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
     *  This method is used to create a {@link Shape} instance from a list of numbers
     *  whose integer values are used to describe the shape of a nd-array.
     *  The shape of an array is the number of elements in each dimension.
     *  @param shape The list of integers which is used to describe the shape of an array.
     *  @return A {@link Shape} instance which is created from the given array of integers.
     */
    static Shape of( List<? extends Number> shape ) {
        return Shape.of(shape.stream().mapToInt(Number::intValue).toArray());
    }

    /**
     *  This method is used to create a {@link Shape} instance from a stream of numbers
     *  whose integer values are used to describe the shape of a nd-array.
     *  The shape of an array is the number of elements in each dimension.
     *  @param shape The stream of integers which is used to describe the shape of an array.
     *  @return A {@link Shape} instance which is created from the given array of integers.
     */
    static Shape of( Stream<? extends Number> shape ) {
        return Shape.of(shape.mapToInt(Number::intValue).toArray());
    }

    /**
     *  This method is used to create a {@link Shape} instance from an iterable of numbers
     *  whose integer values are used to describe the shape of a nd-array.
     *  The shape of an array is the number of elements in each dimension.
     *  @param shape The iterable of integers which is used to describe the shape of an array.
     *  @return A {@link Shape} instance which is created from the given array of integers.
     */
    static Shape of( Iterable<? extends Number> shape ) {
        List<Integer> list = new java.util.ArrayList<>();
        shape.forEach( n -> list.add( n.intValue() ) );
        return Shape.of( list );
    }

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
     * @return The number of elements in the shape.
     */
    default int elements() {
        int elements = 1;
        for ( int i = 0; i < size(); i++ ) elements *= get(i);
        return elements;
    }

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

    /**
     * @return This shape as a stream of integers.
     */
    default Stream<Integer> stream() { return java.util.stream.IntStream.range(0, size()).mapToObj(this::get); }

    /**
     * @param start The start index of the slice, inclusive.
     * @param end The end index of the slice, exclusive.
     * @return A slice of this shape starting at the given start index and ending at the given end index.
     */
    default Shape slice( int start, int end ) {
        if ( start < 0 || end > size() || start > end ) throw new IndexOutOfBoundsException();
        int[] arr = new int[ end - start ];
        for ( int i = start; i < end; i++ ) arr[ i - start ] = get(i);
        return Shape.of( arr );
    }

    /**
     * @param start The start index of the slice, inclusive.
     * @return A slice of this shape starting at the given start index and ending at the end of the shape.
     */
    default Shape slice( int start ) { return slice( start, size() ); }

    /**
     * @param predicate The predicate which is used to filter the shape.
     * @return A new shape which is the result of filtering this shape with the given predicate.
     */
    default Shape filter( java.util.function.Predicate<Integer> predicate ) {
        int[] arr = new int[ size() ];
        int i = 0;
        for ( int j = 0; j < size(); j++ )
            if ( predicate.test( get(j) ) ) arr[i++] = get(j);
        return Shape.of( java.util.Arrays.copyOf( arr, i ) );
    }

    /**
     * @param predicate The predicate which is used to count the elements of the shape for which it is true.
     * @return The number of elements in the shape which satisfy the given predicate.
     */
    default int count( java.util.function.Predicate<Integer> predicate ) {
        int count = 0;
        for ( int i = 0; i < size(); i++ )
            if ( predicate.test( get(i) ) ) count++;
        return count;
    }

    /**
     * @param predicate The predicate which is used to test the elements of the shape.
     * @return True if the given predicate is true for all elements of the shape.
     */
    default boolean every( java.util.function.Predicate<Integer> predicate ) {
        for ( int i = 0; i < size(); i++ )
            if ( !predicate.test( get(i) ) ) return false;
        return true;
    }

    /**
     * @param predicate The predicate which is used to test the elements of the shape.
     * @return True if the given predicate is true for at least one element of the shape.
     */
    default boolean any( java.util.function.Predicate<Integer> predicate ) {
        for ( int i = 0; i < size(); i++ )
            if ( predicate.test( get(i) ) ) return true;
        return false;
    }

    /**
     * @return An iterator over the shape.
     */
    @Override
    default java.util.Iterator<Integer> iterator() {
        return new java.util.Iterator<Integer>() {
            int i = 0;
            @Override public boolean hasNext() { return i < size(); }
            @Override public Integer next() { return get(i++); }
        };
    }

}
