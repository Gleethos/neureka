package neureka.framing;

import lombok.Getter;
import lombok.experimental.Accessors;
import neureka.Component;
import neureka.Tsr;
import java.util.*;
import java.util.function.Function;

@Accessors( prefix = {"_"} )
public class IndexAlias<ValueType> implements Component<Tsr<ValueType>>
{
    private final List<Object> _hiddenKeys = new ArrayList<>();

    @Getter
    private Map<Object, Object> _mapping;
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

    public IndexAlias( Map<Object, List<Object>> labels, Tsr<ValueType> host, String tensorName ) {
        _tensorName = tensorName;
        _mapping = new LinkedHashMap<>( labels.size() * 3 );
        int[] index = { 0 };
        labels.forEach( ( k, v ) -> {
            if ( v != null ) {
                Map<Object, Integer> idxmap = new LinkedHashMap<>( v.size() * 3 );
                for ( int i = 0; i < v.size(); i++ ) idxmap.put( v.get( i ), i );
                if ( !k.equals( index[ 0 ] ) ) _hiddenKeys.add( index[ 0 ] );
                _mapping.put( k, idxmap );
                _mapping.put( index[ 0 ], idxmap ); // default integer index should also always work!
            }
            else {
                if ( !k.equals( index[ 0 ] ) ) _hiddenKeys.add( index[ 0 ] );
                _mapping.put( k, host.getNDConf().shape()[ index[ 0 ] ] );
                _mapping.put( index[ 0 ], host.getNDConf().shape()[ index[ 0 ] ] );// default integer index should also always work!
            }
            index[ 0 ]++;
        });
    }

    public int[] get(List<Object> keys) {
        return get( keys.toArray( new Object[ keys.size() ] ) );
    }

    public int[] get(Object[] keys) {//Todo: iterate over _mapping
        int[] idx = new int[ keys.length ];//_mapping.length];
        for( int i = 0; i < idx.length; i++ ) {
            Object am =  _mapping.get( i );
            if ( am instanceof Map ) {
                idx[ i ] = ( (Map<Object, Integer>) am ).get( keys[ i ] );
            } else if ( am instanceof Integer ) {

            }
            //idx[ i ] = _mapping.get( i ).get(keys[ i ]);
        }
        return idx;
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

    private String _fixed( String str, int size ) {
        if ( str.length() < size ) {
            int first = size / 2;
            int second = size - first;
            first -= str.length() / 2;
            second -= str.length() - str.length() / 2;
            StringBuilder strBuilder = new StringBuilder(str);
            for ( int i = 0; i < first; i++ ) strBuilder.insert(0, " ");
            strBuilder.append( String.join("", Collections.nCopies(Math.max( 0, second ), " ")) );
            //" ".repeat( Math.max( 0, second ) ) );
            return strBuilder.toString();
        }
        return str;
    }


    @Override
    public String toString()
    {
        final int WIDTH = 16;
        final String WALL = " | ";
        final String HEADLINE = "=";
        final String ROWLINE = "-";
        final String CROSS = "+";

        int indexShift = WALL.length() / 2;
        int crossMod = WIDTH+WALL.length();
        Function<Integer, Boolean> isCross = i -> ( i - indexShift ) % crossMod == 0;
        StringBuilder builder = new StringBuilder();
        builder.append( WALL );

        int[] axisLabelSizes = new int[ _mapping.size() ];
        int[] axisCounter = { 0 };
        _mapping.forEach( ( k, v ) -> {
            if ( !_hiddenKeys.contains( k ) ) {
                String axisHeader = k.toString();
                axisHeader = _fixed(axisHeader, WIDTH);
                axisLabelSizes[axisCounter[0]] = axisHeader.length();
                builder.append(axisHeader);
                builder.append(WALL);
                axisCounter[0]++;
            }
        });
        int lineLength = builder.length();
        builder.append( "\n" );
        for ( int i = 0; i < lineLength; i++ ) builder.append( ( isCross.apply( i ) ) ? CROSS : HEADLINE );
        builder.append( "\n" );
        boolean[] hasMoreIndexes = { true };
        int[] depth = { 0 };
        while ( hasMoreIndexes[ 0 ] ) {
            axisCounter[ 0 ] = 0;
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
                        builder.append(_fixed((keyOfDepth[0]).toString(), WIDTH));
                    } else {
                        builder.append(_fixed("---", WIDTH));
                    }
                    builder.append(WALL);
                    axisCounter[0]++;
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
    public void update( Tsr<ValueType> oldOwner, Tsr<ValueType> newOwner ) {
        // This component does not have anything to do when switching owner...
    }
}
