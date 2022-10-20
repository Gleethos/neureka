package neureka.framing;

import neureka.Tsr;
import neureka.common.composition.Component;
import neureka.common.utility.LogUtil;
import neureka.framing.fluent.AxisFrame;

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
 *  Lets for example imagine a tensor of rank 2 with the shape (3, 4), then the axis could for example be labeled
 *  with a tuple of two {@link String} instances like: ("a","b"). <br>
 *  Labeling the indices of the axis for this example requires 2 arrays whose length matches the axis sizes. <br>
 *  The following mapping would be able to label both the axis and their indices: <br>
 *                                                                             <br>
 *  "a" : ["first", "second", "third"],                                        <br>
 *  "b" : ["one", "two", "three", "four"]                                      <br>
 *                                                                             <br>
 *
 * @param <V> The type parameter of the value type of the tensor type to whom this component should belong.
 */
public final class NDFrame<V> implements Component<Tsr<V>>
{
    private final List<Object> _hiddenKeys = new ArrayList<>();
    /**
     *  This {@link Map} contains all the aliases for axis as well as individual
     *  positions for a given axis (in the form of yet another {@link Map}).
     */
    private final Map<Object, Object> _mapping;
    /**
     *  A frame can also carry a name.
     *  When loading a CSV file for example the label would be the first cell if
     *  both index and header labels are included in the file.
     */
    private final String _mainLabel;

    public NDFrame( List<List<Object>> labels, Tsr<V> host, String mainLabel ) {
        this(Collections.emptyMap(), host, mainLabel);
        _label(labels);
    }

    private NDFrame<V> _label( List<List<Object>> labels ) {
        for ( int i = 0; i < labels.size(); i++ ) _mapping.put( i, new LinkedHashMap<>() );
        for ( int i = 0; i < labels.size(); i++ ) {
            if ( labels.get( i ) != null ) {
                for ( int j = 0; j < labels.get( i ).size(); j++ ) {
                    if ( labels.get( i ).get( j ) != null )
                        atAxis( i ).atIndexAlias( labels.get( i ).get( j ) ).setIndex( j );
                }
            }
        }
        return this;
    }

    public NDFrame( Tsr<V> host, String tensorName ) {
        this(Collections.emptyMap(), host, tensorName);
    }

    public NDFrame(Map<Object, List<Object>> labels, Tsr<V> host, String tensorName ) {
        _mainLabel = tensorName;
        _mapping = new LinkedHashMap<>( labels.size() * 3 );
        for ( int i = 0; i < host.rank(); i++ ) {
            Map<Object, Object> indicesMap = new LinkedHashMap<>();
            //indicesMap.put( i, i );
            _mapping.put( i, indicesMap );
        }
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

    private NDFrame( List<Object> hiddenKeys, Map<Object, Object> mapping, String tensorName ) {
        _hiddenKeys.addAll( hiddenKeys );
        _mapping = new LinkedHashMap<>(mapping);
        _mainLabel = tensorName;
    }

    public NDFrame<V> withLabel( String newLabel ) {
        return new NDFrame<>( _hiddenKeys, _mapping, newLabel );
    }

    public NDFrame<V> withAxesLabels( List<List<Object>> labels ) {
        return new NDFrame<V>( _hiddenKeys, _mapping, _mainLabel )._label(labels);
    }

    public int[] get( List<Object> keys ) {
        LogUtil.nullArgCheck( keys, "keys", List.class );
        return get( keys.toArray( new Object[0] ) );
    }

    public int[] get( Object... keys ) {//Todo: iterate over _mapping
        LogUtil.nullArgCheck( keys, "keys", Object[].class );
        int[] indices = new int[ keys.length ];
        for( int i = 0; i < indices.length; i++ ) {
            Object am = _mapping.get( i );
            if ( am instanceof Map )
                indices[ i ] = ( (Map<Object, Integer>) am ).get( keys[ i ] );
            else if ( am instanceof Integer )
                indices[ i ] = (Integer) am;
        }
        return indices;
    }

    public boolean hasLabelsForAxis( Object axisAlias ) {
        LogUtil.nullArgCheck( axisAlias, "axisAlias", Object.class );
        return !atAxis(axisAlias).getAllAliases().isEmpty();
    }

    /**
     *  A {@link NDFrame} exposes aliases for axes as well as aliases for individual positions within an axis.
     *  This method returns a view on a axis which is targeted by an axis alias as key.
     *  This view is an instance of the {@link AxisFrame} class which provides useful methods
     *  for getting or setting alias objects for individual positions for the given axis.
     *  This is useful when for example replacing certain aliases or simply taking a look at them.
     *
     * @param axisAlias The axis alias object which targets an {@link AxisFrame} of {@link NDFrame}.
     * @return A view of the targeted axis in the for of an{@link AxisFrame} which provides getters and setters for aliases.
     */
    public AxisFrame<Integer, V> atAxis( Object axisAlias )
    {
        LogUtil.nullArgCheck( axisAlias, "axisAlias", Object.class );
        return AxisFrame.<Integer, Integer, V>builder()
                .getter(
                        atKey -> () ->
                        {
                            Object am =  _mapping.get( axisAlias );
                            if ( am instanceof Map )
                                return ((Map<Object, Integer>) _mapping.get( axisAlias )).get(atKey);
                            else
                                return 0;
                        }
                )
                .setter(
                        atKey -> (int setValue) ->
                        {
                            Map<Object, Integer> am = _initializeIdxmap( axisAlias, atKey, setValue );
                            am.put( atKey, setValue );
                            return this;
                        }
                )
                .replacer(
                        ( currentIndexKey ) -> (newIndexKey ) -> {
                            Map<Object, Integer> am = _initializeIdxmap( axisAlias, currentIndexKey, (Integer) currentIndexKey );
                            if (am.containsKey( currentIndexKey ) )
                                am.put( newIndexKey, am.remove( currentIndexKey ) ); // This...
                            return this;
                        }
                )
                .allAliasGetter(
                        () -> {
                            Object am =  _mapping.get( axisAlias );
                            if ( am == null ) return new ArrayList<>();
                            List<Object> keys = new ArrayList<>();
                            if ( am instanceof Map ) ( (Map<Object, Integer>) am ).forEach( ( k, v ) -> keys.add( k ) );
                            else for ( int i = 0; i < ( (Integer) am ); i++ ) keys.add( i );
                            return keys;
                        }
                )
                .allAliasGetterFor(
                        (index) -> {
                            List<Object> keys = new ArrayList<>();
                            Object am =  _mapping.get( axisAlias );
                            if ( am instanceof Map ) ( (Map<Object, Integer>) am ).forEach( (k, v) -> { if (v.equals(index)) keys.add( k ); } );
                            else keys.add( index );
                            return keys;
                        }
                )
                .build();
    }

    private Map<Object, Integer> _initializeIdxmap( Object axis, Object key, int index ) {
        Object am =  _mapping.get( axis );
        if ( am instanceof Map )
            return (Map<Object, Integer>) _mapping.get( axis );

        int size = (Integer) am;

        Map<Object, Integer> newIdxmap = new LinkedHashMap<>( size * 3 );
        for( int i = 0; i < size; i++ ) {
            if ( index == i ) newIdxmap.put( key, i );
            else newIdxmap.put( i, i );
        }
        _mapping.put( axis, newIdxmap );
        return newIdxmap;
    }

    /**
     *  This method simply pads the provided string based on the size passed to it.
     *
     * @param string The {@link String} which ought to be padded by white spaces.
     * @param cellSize The length of the padded {@link String} which will be returned.
     * @return The padded {@link String}.
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
                            if ( iv == depth[0] ) keyOfDepth[0] = ik;
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
    public boolean update( OwnerChangeRequest<Tsr<V>> changeRequest ) {
        changeRequest.executeChange(); // This can be an 'add', 'remove' or 'transfer' of this component!
        // This component does not have anything to do when switching owner...
        return true;
    }

    public Map<Object, Object> getMapping() {
        return _mapping;
    }

    public String getLabel() {
        return _mainLabel;
    }
}
