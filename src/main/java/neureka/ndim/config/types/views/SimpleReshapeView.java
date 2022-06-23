package neureka.ndim.config.types.views;

import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

public class SimpleReshapeView extends AbstractNDC
{
    private final NDConfiguration _toBeViewed;
    private final int[] _form;
    private final int[] _formTranslator;
    private final int[] _shape;
    private final int[] _translation;
    private final int[] _indicesMap;
    private final int[] _spread;
    private final int[] _offset;

    public SimpleReshapeView( int[] form, NDConfiguration toBeViewed )
    {
        _toBeViewed = toBeViewed;
        _form = form;

        List<Integer> _translator = new ArrayList<>();

        for ( int i=0; i < form.length; i++ )
                if ( form[ i ] < -1 ) throw new IllegalArgumentException(
                    "SimpleReshapeView may not view a NDConfiguration beyond reshaping and or padding!"
                );
            else if ( form[ i ] >= 0 )  _translator.add( form[ i ] );

        _formTranslator = _translator.stream().mapToInt( e -> e ).toArray();


        NDConfiguration ndc = _simpleReshape( form, toBeViewed );
        _shape = ndc.shape();
        _translation = ndc.translation();
        _indicesMap = ndc.indicesMap();
        _spread = ndc.spread();
        _offset = ndc.offset();
    }

    /** {@inheritDoc} */
    @Override public final int rank() { return _shape.length; }

    /** {@inheritDoc} */
    @Override public final int[] shape() { return _shape; }

    /** {@inheritDoc} */
    @Override public final int shape( int i ) { return _shape[ i ]; }

    /** {@inheritDoc} */
    @Override public final int[] indicesMap() { return _indicesMap; }

    /** {@inheritDoc} */
    @Override public final int indicesMap(int i ) { return _indicesMap[ i ]; }

    /** {@inheritDoc} */
    @Override public final int[] translation() { return _translation; }

    /** {@inheritDoc} */
    @Override public final int translation( int i ) { return _translation[ i ]; }

    /** {@inheritDoc} */
    @Override public final int[] spread() { return _spread; }

    /** {@inheritDoc} */
    @Override public final int spread( int i ) { return _spread[ i ]; }

    /** {@inheritDoc} */
    @Override public final int[] offset() { return _offset; }

    /** {@inheritDoc} */
    @Override public final int offset( int i ) { return _offset[ i ]; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndex( int index ) { return indexOfIndices( indicesOfIndex(index) ); }

    /** {@inheritDoc} */
    @Override
    public final int[] indicesOfIndex( int index ) {
        int[] indices = new int[ _shape.length ];
        for ( int ii = 0; ii < rank(); ii++ ) {
            indices[ ii ] += index / _indicesMap[ ii ];
            index %= _indicesMap[ ii ];
        }
        return indices;
    }

    /** {@inheritDoc} */
    @Override
    public final int indexOfIndices( int[] indices ) {
        int[] innerIdx = _rearrange( indices, _form, _formTranslator );
        return _toBeViewed.indexOfIndices( innerIdx );
    }

    @Contract(pure = true)
    private static int[] _rearrange( @NotNull int[] array, @NotNull int[] ptr, @NotNull int[] indices ) {
        for ( int i = 0; i < ptr.length; i++ ) {
            if ( ptr[ i ] >= 0 ) indices[ ptr[ i ] ] = array[ i ];
        }
        return indices;
    }

}
