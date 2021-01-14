package neureka.ndim.config.types.views;

import neureka.Neureka;
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
    private final int[] _idxmap;
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
        _idxmap = ndc.idxmap();
        _spread = ndc.spread();
        _offset = ndc.offset();
    }

    @Override
    public int rank() {
        return _shape.length;
    }

    @Override
    public int[] shape() {
        return _shape;
    }

    @Override
    public int shape( int i ) {
        return _shape[ i ];
    }

    @Override
    public int[] idxmap() {
        return _idxmap;
    }

    @Override
    public int idxmap( int i ) {
        return _idxmap[ i ];
    }

    @Override
    public int[] translation() {
        return _translation;
    }

    @Override
    public int translation( int i ) {
        return _translation[ i ];
    }

    @Override
    public int[] spread() {
        return _spread;
    }

    @Override
    public int spread( int i ) {
        return _spread[ i ];
    }

    @Override
    public int[] offset() {
        return _offset;
    }

    @Override
    public int offset( int i ) {
        return _offset[ i ];
    }

    @Override
    public int i_of_i( int i ) {
        return i_of_idx( idx_of_i( i ) );
    }

    @Override
    public int[] idx_of_i( int i ) {
        int[] idx = new int[_shape.length];
        for ( int ii = 0; ii < rank(); ii++ ) {
            idx[ ii ] += i / _idxmap[ ii ];
            i %= _idxmap[ ii ];
        }
        return idx;
    }

    @Override
    public int i_of_idx( int[] idx ) {
        int[] innerIdx = _rearrange( idx,_form, _formTranslator );
        return _toBeViewed.i_of_idx( innerIdx );
    }

    @Contract(pure = true)
    private static int[] _rearrange(@NotNull int[] array, @NotNull int[] ptr, @NotNull int[] idx) {
        for ( int i = 0; i < ptr.length; i++ ) {
            if ( ptr[ i ] >= 0 ) idx[ptr[ i ]] = array[ i ];
        }
        return idx;
    }

}
