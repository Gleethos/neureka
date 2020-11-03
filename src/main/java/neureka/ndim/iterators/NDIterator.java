package neureka.ndim.iterators;


import neureka.Neureka;
import neureka.Tsr;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.complex.ComplexD1Configuration;
import neureka.ndim.config.types.complex.ComplexD2Configuration;
import neureka.ndim.config.types.complex.ComplexD3Configuration;
import neureka.ndim.config.types.simple.SimpleD1Configuration;
import neureka.ndim.config.types.simple.SimpleD2Configuration;
import neureka.ndim.config.types.simple.SimpleD3Configuration;
import neureka.ndim.config.types.virtual.VirtualNDConfiguration;
import neureka.ndim.iterators.types.*;
import neureka.ndim.iterators.types.complex.ComplexD1CIterator;
import neureka.ndim.iterators.types.complex.legacy.ComplexLegacyD2CIterator;
import neureka.ndim.iterators.types.complex.legacy.ComplexLegacyD3CIterator;
import neureka.ndim.iterators.types.complex.main.ComplexD2CIterator;
import neureka.ndim.iterators.types.complex.main.ComplexD3CIterator;
import neureka.ndim.iterators.types.simple.SimpleD1CIterator;
import neureka.ndim.iterators.types.simple.legacy.SimpleLegacyD2CIterator;
import neureka.ndim.iterators.types.simple.legacy.SimpleLegacyD3CIterator;
import neureka.ndim.iterators.types.simple.main.SimpleD2CIterator;
import neureka.ndim.iterators.types.simple.main.SimpleD3CIterator;

import java.util.StringJoiner;
import java.util.stream.IntStream;

public interface NDIterator
{
    static NDIterator of( Tsr<?> t ){

        NDConfiguration ndc = t.getNDConf();

        if ( ndc instanceof ComplexD1Configuration) return new ComplexD1CIterator( (ComplexD1Configuration) ndc );
        if ( ndc instanceof SimpleD1Configuration ) return new SimpleD1CIterator( (SimpleD1Configuration) ndc );

        if ( Neureka.instance().settings().indexing().isUsingLegacyIndexing() ) {
            if ( ndc instanceof ComplexD2Configuration) return new ComplexLegacyD2CIterator( (ComplexD2Configuration) ndc );
            if ( ndc instanceof ComplexD3Configuration) return new ComplexLegacyD3CIterator( (ComplexD3Configuration) ndc );
            if ( ndc instanceof SimpleD2Configuration ) return new SimpleLegacyD2CIterator( (SimpleD2Configuration) ndc );
            if ( ndc instanceof SimpleD3Configuration ) return new SimpleLegacyD3CIterator( (SimpleD3Configuration) ndc );
        } else {
            if ( ndc instanceof ComplexD2Configuration) return new ComplexD2CIterator( (ComplexD2Configuration) ndc );
            if ( ndc instanceof ComplexD3Configuration) return new ComplexD3CIterator( (ComplexD3Configuration) ndc );
            if ( ndc instanceof SimpleD2Configuration ) return new SimpleD2CIterator( (SimpleD2Configuration) ndc );
            if ( ndc instanceof SimpleD3Configuration ) return new SimpleD3CIterator( (SimpleD3Configuration) ndc );
        }

        if ( ndc instanceof VirtualNDConfiguration )
            return new VirtualNDIterator( (VirtualNDConfiguration) ndc );
        else
            return new DefaultNDIterator( ndc );
    }

    int shape( int i );

    int[] shape();

    void increment();

    void decrement();

    int i();

    int get( int axis );

    int[] get();

    void set( int axis, int position );

    void set( int[] idx );

    int rank();


    default String asString()
    {
        StringBuilder b = new StringBuilder();

        StringJoiner sj = new StringJoiner( "," );
        StringJoiner finalSj1 = sj;
        IntStream.of( this.shape() ).forEach( x -> finalSj1.add( String.valueOf(x) ) );

        b.append( "S[" + sj.toString() + "];" );
        sj = new StringJoiner( "," );
        StringJoiner finalSj = sj;
        IntStream.of( this.get() ).forEach( x -> finalSj.add( String.valueOf( x ) ) );
        b.append( "I[" + sj.toString() + "];" );
        return b.toString();
    }


}
