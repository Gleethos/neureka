package neureka.ndim.config;


import neureka.Neureka;
import neureka.Tsr;
import neureka.ndim.config.types.D1C;
import neureka.ndim.config.types.D2C;
import neureka.ndim.config.types.D3C;
import neureka.ndim.config.types.virtual.VirtualNDConfiguration;
import neureka.ndim.iterators.*;
import neureka.ndim.iterators.legacy.LegacyD2Iterator;
import neureka.ndim.iterators.legacy.LegacyD3Iterator;

public interface NDIterator
{
    static NDIterator of( Tsr<?> t ){

        NDConfiguration ndc = t.getNDConf();

        if ( ndc instanceof D1C ) return new D1Iterator( (D1C) ndc );
        else if ( ndc instanceof D2C )
            return ( Neureka.instance().settings().indexing().isUsingLegacyIndexing() )
                    ? new LegacyD2Iterator( (D2C) ndc )
                    : new D2Iterator( (D2C) ndc );
        else if ( ndc instanceof D3C )
            return ( Neureka.instance().settings().indexing().isUsingLegacyIndexing() )
                    ? new LegacyD3Iterator( (D3C) ndc )
                    : new D3Iterator( (D3C) ndc );
        else if ( ndc instanceof VirtualNDConfiguration )
            return new VirtualNDIterator( (VirtualNDConfiguration) ndc );
        else
            return new DefaultNDIterator( ndc );

    }


    int[] shape();

    void increment();

    void decrement();

    int i();

    int get( int axis );

    int[] get();

    void set( int axis, int position );

    void set( int[] idx );

    int rank();
}
