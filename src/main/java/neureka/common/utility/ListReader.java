package neureka.common.utility;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *  This is a simple utility class which traverses nested data structures and converts them into
 *  information which can be used to instantiate a tensor,
 *  namely: A flat data array, a shape array and a type class.
 */
public final class ListReader
{
    private final Class<?> _type;
    private final int _size;

    /**
     *  Reads the provided data and turns it into a {@link Result} object,
     *  containing a flattened list of the data alongside its
     *  shape and data type.
     *
     * @param data A list of data elements or nested lists with an arbitrary degree of nesting.
     * @param valueFilter A filter for the elements in the provided data list.
     * @return The result object containing data, data type and shape information.
     */
    public static Result read( List<Object> data, Function<Object, Object> valueFilter ) {
        return new Result( data, valueFilter );
    }

    private ListReader(
            Object data,
            int depth,
            List<Object> growingData,
            List<Integer> growingShape,
            Function<Object, Object> valueFilter
    ) {

        List<ListReader> readers;
        if ( data instanceof List ) {
            List<Object> list = ((List<Object>) data).stream()
                                                    .map( valueFilter )
                                                    .collect(Collectors.toList());

            long leaves = list.stream().filter(this::_isLeave).count();
            if ( leaves != list.size() && leaves != 0 ) {
                String message = "Inconsistent degree of list nesting encountered at depth " + depth + ".";
                throw new IllegalArgumentException(message);
            }
            if ( growingShape.size() == depth ) growingShape.add(list.size());

            readers = list.stream()
                           .map( o -> new ListReader( o, depth + 1, growingData, growingShape, valueFilter ) )
                           .collect(Collectors.toList());

            _type = _findType(readers);
            _size = _findSize(readers, depth);
        }
        else
        {
            _type = ( data == null ? null : data.getClass() );
            _size = 1;
            growingData.add( data );
        }
    }

    private Class<?> _findType( List<ListReader> readers ) {
        Supplier<Stream<Class<?>>> types = () -> readers.stream().map(r -> r._type );
        Class<?> firstType = types.get().findFirst().orElse(Object.class);
        long numberOfSameType = types.get().filter( t -> t == firstType ).count();
        if ( numberOfSameType != readers.size() ) {
            String message = "Type inconsistency encountered. Not all leave elements are of the same type!\n" +
                                "Expected type '" +
                                            firstType.getSimpleName() +
                                "', but encountered '" +
                                            types.get().filter( t -> t != firstType ).findAny().orElse(Object.class).getSimpleName() +
                                "'.";
            throw new IllegalArgumentException(message);
        }
        return firstType;
    }

    private int _findSize( List<ListReader> readers, int depth ) {
        Supplier<Stream<Integer>> sizes = () -> readers.stream().map(r -> r._size );
        int firstSize = sizes.get().findFirst().orElse(0);
        long numberOfSameSize = sizes.get().filter( s -> s == firstSize ).count();
        if ( numberOfSameSize != readers.size() ) {
            String message = "Size inconsistency encountered at nest level '"+depth+"'. Not all nested lists are equally sized.\n" +
                            "Expected size '"+firstSize+"', but encountered '"+sizes.get().filter( s -> s != firstSize ).findAny().orElse(0)+"'.";
            throw new IllegalArgumentException(message);
        }
        return readers.stream().map( r -> r._size ).reduce( 0, Integer::sum );
    }

    private boolean _isLeave( Object o ) {
        if ( o == null ) return true;
        boolean isList = o instanceof List;
        return isList && ((List<?>) o).isEmpty();
    }

    public static class Result {

        private final List<Integer> _growingShape = new ArrayList<>();
        private final List<Object> _growingData = new ArrayList<>();
        private final Class<?> _type;

        private Result(Object data, Function<Object, Object> valueFilter
        ) {
            ListReader reader = new ListReader(
                                        data,
                                    0,
                                        _growingData,
                                        _growingShape,
                                        valueFilter
                                    );
            _type = reader._type;
        }

        public Class<?> getType() { return _type; }

        public List<Integer> getShape() { return _growingShape; }

        public List<Object> getData() { return _growingData; }

    }

}
