package neureka.framing;

import lombok.Getter;
import lombok.experimental.Accessors;
import neureka.Component;
import neureka.Tsr;
import java.util.*;
import java.util.function.Function;

/**
 *  Instances of this class are components of tensors, which store aliases for the indices of the tensor.
 *  These indices aliases can be anything that has an identity, meaning any plain old object. <br>
 *  There are two layers of aliasing/labeling provided by this class:
 *  <ul>
 *      <li>
 *          Labels for the axis of a tensor, which are the indices of its shape array.
 *      </li>
 *      <li>
 *          Labels for the indices of a specific axis.
 *      </li>
 *  </ul>
 *  Let's for example imagine a tensor of rank 2 with the shape (3, 4), then the axis could for example be labeled
 *  with a tuple of two String instances like: ("a","b"). <br>
 *  Labeling the indices of the axis for this example requires 2 arrays whose length matches the axis sizes. <br>
 *  The following mapping would be able to label both the axis as well as their indices: <br>
 *                                                                             <br>
 *  "a" : ["first", "second", "third"],                                <br>
 *  "b" : ["one", "two", "three", "four"]                              <br>
 *                                                                     <br>
 *
 * @param <ValType> The type parameter of the value type of the tensor type to whom this component should belong.
 */
@Accessors( prefix = {"_"} )
public final class IndexAlias<ValType> implements Component<Tsr<ValType>>
{
    private final List<Object> _hiddenKeys = new ArrayList<>();

    @Getter
    private final Map<Object, Object> _mapping;
    @Getter
    private final String _tensorName;

    public IndexAlias( List<List<Object>> labels, String tensorName ) {
        _tensorName = tensorName;
        _mapping = new LinkedHashMap<>(labels.size());
        for ( int i = 0; i < labels.size(); i++ ) _mapping.put( i, new LinkedHashMap<>() );
        for ( int i = 0; i < labels.size(); i++ ) {
            if ( labels.get( i ) != null ) {
                for ( int ii = 0; ii < labels.get( i ).size(); ii++ ) {
                    if ( labels.get( i ).get( ii ) != null ) set( i, labels.get( i ).get( ii ), ii );
                }
            }
        }
    }

    public IndexAlias( int size, String tensorName ) {
        _tensorName = tensorName;
        _mapping = new LinkedHashMap<>( size );
        for ( int i = 0; i < size; i++ ) _mapping.put( i, new LinkedHashMap<>() );
    }

    public IndexAlias( Map<Object, List<Object>> labels, Tsr<ValType> host, String tensorName ) {
        _tensorName = tensorName;
        _mapping = new LinkedHashMap<>( labels.size() * 3 );
        int[] index = { 0 };
        labels.forEach( ( k, v ) -> {
            if ( v != null ) {
                Map<Object, Integer> indicesMap = new LinkedHashMap<>( v.size() * 3 );
                for ( int i = 0; i < v.size(); i++ ) indicesMap.put( v.get( i ), i );
                if ( !k.equals( index[ 0 ] ) ) _hiddenKeys.add( index[ 0 ] );
                _mapping.put( k, indicesMap );
                _mapping.put( index[ 0 ], indicesMap ); // default integer index should also always work!
            }
            else {
                if ( !k.equals( index[ 0 ] ) ) _hiddenKeys.add( index[ 0 ] );
                _mapping.put( k, host.getNDConf().shape()[ index[ 0 ] ] );
                _mapping.put( index[ 0 ], host.getNDConf().shape()[ index[ 0 ] ] );// default integer index should also always work!
            }
            index[ 0 ]++;
        });
    }

    public int[] get( List<Object> keys ) {
        return get( keys.toArray( new Object[ keys.size() ] ) );
    }

    public int[] get( Object[] keys ) {//Todo: iterate over _mapping
        int[] indices = new int[ keys.length ];//_mapping.length];
        for( int i = 0; i < indices.length; i++ ) {
            Object am =  _mapping.get( i );
            if ( am instanceof Map ) {
                indices[ i ] = ( (Map<Object, Integer>) am ).get( keys[ i ] );
            } else if ( am instanceof Integer ) {
                // TODO: Implement
            }
        }
        return indices;
    }

    public int get( Object key, Object axis )
    {
        Object am =  _mapping.get( axis );
        if ( am instanceof Map ) return ( (Map<Object, Integer>) am ).get( key );
        return ( key instanceof Integer ) ? ( (Integer) key ) : 0;
    }

    public void replace( Object axis, Object indexKey, Object newIndexKey )
    {
        Object am =  _mapping.get(axis);
        if ( am instanceof Map )
            ( (Map<Object, Integer>) am ).put( newIndexKey, ((Map<Object, Integer>) am ).remove( indexKey ) );
        else
            if ( am instanceof Integer ) _initializeIdxmap( axis, (Integer) am, newIndexKey, (Integer) indexKey );
    }

    private void _initializeIdxmap( Object axis, int size, Object key, int index ) {
        Map<Object, Integer> newIdxmap = new LinkedHashMap<>( size * 3 );
        for( int i = 0; i < size; i++ ) {
            if ( index == i ) newIdxmap.put( key, i );
            else newIdxmap.put( i, i );
        }
        _mapping.put( axis, newIdxmap );
    }

    public void set( Object axis, Object indexKey, int index )
    {
        Object am =  _mapping.get( axis );
        if ( am instanceof Map ) ( (Map<Object, Integer>) am ).put( indexKey, index );
        else if ( am instanceof Integer ) _initializeIdxmap( axis, (Integer) am, indexKey, index );
    }

    public List<Object> keysOf( Object axis )
    {
        Object am =  _mapping.get( axis );
        if ( am == null ) return null;
        List<Object> keys = new ArrayList<>();
        if ( am instanceof Map ) ( (Map<Object, Integer>) am ).forEach( ( k, v ) -> keys.add( k ) );
        else for ( int i = 0; i < ( (Integer) am ); i++ ) keys.add( i );
        return keys;
    }

    public List<Object> keysOf( Object axis, int index ) {
        List<Object> keys = new ArrayList<>();
        Object am =  _mapping.get( axis );
        if ( am instanceof Map ) ( (Map<Object, Integer>) am ).forEach( (k, v) -> { if ( v == index ) keys.add( k ); } );
        else keys.add( index );
        return keys;
    }

    /**
     *  This method simply pads the provided string based on the size passed to it.
     *
     * @param string
     * @param cellSize
     * @return
     */
    private String _paddedCentered( String string, int cellSize ) {
        if ( string.length() < cellSize ) {
            int first = cellSize / 2;
            int second = cellSize - first;
            first -= string.length() / 2;
            second -= string.length() - string.length() / 2;
            StringBuilder strBuilder = new StringBuilder(string);
            // Now we prepend the prefix spaces:
            for ( int i = 0; i < first; i++ ) strBuilder.insert(0, " ");
            strBuilder.append( String.join("", Collections.nCopies(Math.max( 0, second ), " ")) );
            // ...equal to the following expression:  " ".repeat( Math.max( 0, second ) ) );
            return strBuilder.toString();
        }
        return string;
    }


    @Override
    public String toString()
    {
        final int TABLE_CELL_WIDTH = 16;
        final String WALL = " | ";
        final String HEADLINE = "=";
        final String ROWLINE = "-";
        final String CROSS = "+";

        int indexShift = WALL.length() / 2;
        int crossMod = TABLE_CELL_WIDTH+WALL.length();
        Function<Integer, Boolean> isCross = i -> ( i - indexShift ) % crossMod == 0;
        StringBuilder builder = new StringBuilder();
        builder.append( WALL );

        _mapping.forEach( ( k, v ) -> {
            if ( !_hiddenKeys.contains( k ) ) {
                String axisHeader = k.toString();
                axisHeader = _paddedCentered(axisHeader, TABLE_CELL_WIDTH);
                builder.append(axisHeader);
                builder.append(WALL);
            }
        });
        int lineLength = builder.length();
        builder.append( "\n" );
        for ( int i = 0; i < lineLength; i++ ) builder.append( ( isCross.apply( i ) ) ? CROSS : HEADLINE );
        builder.append( "\n" );
        boolean[] hasMoreIndexes = { true };
        int[] depth = { 0 };
        while ( hasMoreIndexes[ 0 ] ) {
            Object[] keyOfDepth = { null };
            builder.append( WALL );
            _mapping.forEach( ( k, v ) -> {
                if ( !_hiddenKeys.contains( k ) ) {
                    keyOfDepth[0] = null;
                    if (v instanceof Map) {
                        ((Map<Object, Integer>) v).forEach((ik, iv) -> {
                            if (iv.intValue() == depth[0]) keyOfDepth[0] = ik;
                        });
                    } else if (v instanceof Integer) {
                        if (depth[0] < ((Integer) v)) keyOfDepth[0] = depth[0];
                    }
                    if (keyOfDepth[0] != null) {
                        builder.append(_paddedCentered((keyOfDepth[0]).toString(), TABLE_CELL_WIDTH));
                    } else {
                        builder.append(_paddedCentered("---", TABLE_CELL_WIDTH));
                    }
                    builder.append(WALL);
                }
            });
            depth[ 0 ]++;
            builder.append( "\n" );
            for( int i = 0; i < lineLength; i++ ) builder.append( ( isCross.apply( i ) ) ? CROSS : ROWLINE );
            builder.append( "\n" );
            if ( keyOfDepth[ 0 ] == null ) hasMoreIndexes[ 0 ] = false;
        }

        StringBuilder result = new StringBuilder().append( "\nTensor IndexAlias: axis/indexes" );
        result.append( "\n" );
        for ( int i = 0; i < lineLength; i++ ) result.append( HEADLINE );
        result.append( "\n" );

        result.append( builder );
        return result.toString();
    }


    @Override
    public void update( Tsr<ValType> oldOwner, Tsr<ValType> newOwner ) {
        // This component does not have anything to do when switching owner...
    }
}
