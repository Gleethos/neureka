/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   _   _ _____ _____ _                 _
  | \ | |  __ \_   _| |               | |
  |  \| | |  | || | | |_ ___ _ __ __ _| |_ ___  _ __
  | . ` | |  | || | | __/ _ \ '__/ _` | __/ _ \| '__|
  | |\  | |__| || |_| ||  __/ | | (_| | || (_) | |
  |_| \_|_____/_____|\__\___|_|  \__,_|\__\___/|_|


*/


package neureka.ndim.iterators;

import neureka.Tsr;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.complex.ComplexD1Configuration;
import neureka.ndim.config.types.complex.ComplexD2Configuration;
import neureka.ndim.config.types.complex.ComplexD3Configuration;
import neureka.ndim.config.types.simple.SimpleD1Configuration;
import neureka.ndim.config.types.simple.SimpleD2Configuration;
import neureka.ndim.config.types.simple.SimpleD3Configuration;
import neureka.ndim.config.types.virtual.VirtualNDConfiguration;
import neureka.ndim.iterators.types.DefaultNDIterator;
import neureka.ndim.iterators.types.VirtualNDIterator;
import neureka.ndim.iterators.types.complex.ComplexD1CIterator;
import neureka.ndim.iterators.types.complex.main.ComplexD2CIterator;
import neureka.ndim.iterators.types.complex.main.ComplexD3CIterator;
import neureka.ndim.iterators.types.simple.SimpleD1CIterator;
import neureka.ndim.iterators.types.simple.main.SimpleD2CIterator;
import neureka.ndim.iterators.types.simple.main.SimpleD3CIterator;

import java.util.StringJoiner;
import java.util.stream.IntStream;

public interface NDIterator
{
    static NDIterator of( Tsr<?> t ) {

        NDConfiguration ndc = t.getNDConf();

        if ( ndc instanceof ComplexD1Configuration) return new ComplexD1CIterator( (ComplexD1Configuration) ndc );
        if ( ndc instanceof SimpleD1Configuration ) return new SimpleD1CIterator( (SimpleD1Configuration) ndc );

        if ( ndc instanceof ComplexD2Configuration) return new ComplexD2CIterator( (ComplexD2Configuration) ndc );
        if ( ndc instanceof ComplexD3Configuration) return new ComplexD3CIterator( (ComplexD3Configuration) ndc );
        if ( ndc instanceof SimpleD2Configuration ) return new SimpleD2CIterator( (SimpleD2Configuration) ndc );
        if ( ndc instanceof SimpleD3Configuration ) return new SimpleD3CIterator( (SimpleD3Configuration) ndc );

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

    void set( int[] indices );

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
