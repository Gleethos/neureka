package neureka.utility;

import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ListReader {

    private final Class<?> _type;
    private final List<ListReader> _readers;
    private final int _size;

    public ListReader(
            Object data,
            int depth,
            List<Object> growingData,
            List<Integer> growingShape,
            Function<Object, Object> valueFilter
    ) {

        if ( data instanceof List ) {
            List<Object> list = ((List<Object>) data).stream()
                                                    .map( valueFilter )
                                                    .collect(Collectors.toList());

            long leaves = list.stream().filter( o -> _isLeave(o) ).count();
            if ( leaves != list.size() && leaves != 0 ) {
                String message = "Inconsistent degree of list nesting encountered at depth " + depth + ".";
                throw new IllegalArgumentException(message);
            }
            if ( growingShape.size() == depth ) growingShape.add(list.size());
            _readers = list.stream()
                           .map( o -> new ListReader( o, depth + 1, growingData, growingShape, valueFilter ) )
                           .collect(Collectors.toList());

            _type = _findType(_readers);
            _size = _findSize(_readers, depth);
        }
        else
        {
            _type = ( data == null ? null : data.getClass() );
            _size = 1;
            _readers = null;
            growingData.add( data );
        }
    }

    public Class<?> getType() { return _type; }

    private Class<?> _findType( List<ListReader> readers ) {
        Supplier<Stream<Class<?>>> types = () -> readers.stream().map(r -> r._type );
        Class<?> firstType = types.get().findFirst().get();
        long numberOfSameType = types.get().filter( t -> t == firstType ).count();
        if ( numberOfSameType != readers.size() ) {
            String message = "Type inconsistency encountered. Not all leave elements are of the same type!\n" +
                                "Expected type '" +
                                            firstType.getSimpleName() +
                                "', but encountered '" +
                                            types.get().filter( t -> t != firstType ).findAny().get().getSimpleName() +
                                "'.";
            throw new IllegalArgumentException(message);
        }
        return firstType;
    }

    private int _findSize( List<ListReader> readers, int depth ) {
        Supplier<Stream<Integer>> sizes = () -> readers.stream().map(r -> r._size );
        int firstSize = sizes.get().findFirst().get();
        long numberOfSameSize = sizes.get().filter( s -> s == firstSize ).count();
        if ( numberOfSameSize != readers.size() ) {
            String message = "Size inconsistency encountered at nest level '"+depth+"'. Not all nested lists are equally sized.\n" +
                            "Expected size '"+firstSize+"', but encountered '"+sizes.get().filter( s -> s != firstSize ).findAny().get()+"'.";
            throw new IllegalArgumentException(message);
        }
        return readers.stream().map( r -> r._size ).reduce( 0, Integer::sum );
    }

    private boolean _isLeave(Object o) {
        if ( o == null ) return true;
        boolean isList = o instanceof List;
        if ( isList && ( (List) o ).isEmpty() ) return true;
        else return false;
    }


}
