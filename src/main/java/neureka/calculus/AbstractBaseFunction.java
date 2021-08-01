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

*/

package neureka.calculus;

import neureka.Tsr;

import java.util.List;

/**
 * This class implements certain methods by simply calling
 * other methods which are responsible for executing
 * the important logic implemented in sub classes.    <br><br>
 *
 * The reason for this is simply that otherwise there
 * would be many redundantly implemented methods.
 * The 'call' and 'invoke' methods with the same arguments
 * are supposed to do the same thing, however
 * they are both part of the {@link Function} interface in order to
 * allow for overloading the '()' operator in different
 * JVM languages...
 */
public abstract class AbstractBaseFunction implements Function
{

    @Override
    public double call( double input ) {
        return call(new double[]{input});
    }

    @Override
    public <T> Tsr<T> call( Tsr<T> input ) {
        return call(new Tsr[]{input});
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public <T> Tsr<T> call( List<Tsr<T>> inputs ) {
        return call(inputs.toArray(new Tsr[ 0 ]));
    }

    @Override
    public <T> Tsr<T> invoke( List<Tsr<T>> inputs ) {
        return call( inputs );
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public double invoke( double input ) {
        return call( input );
    }

    @Override
    public double invoke( double[] inputs, int j ) {
        return call( inputs, j );
    }

    @Override
    public double invoke( double... inputs ) {
        return call( inputs );
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public <T> Tsr<T> invoke( Tsr<T> input ) {
        return call( input );
    }

    @Override
    public <T> Tsr<T> invoke( Tsr<T>[] inputs, int j ) {
        return call( inputs, j );
    }

    @Override
    public <T> Tsr<T> invoke( Tsr<T>... inputs ) {
        return call( inputs );
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public <T> Tsr<T> derive( List<Tsr<T>> inputs, int index, int j ) {
        return derive( inputs.toArray( new Tsr[ 0 ] ), index, j );
    }

    @Override
    public <T> Tsr<T> derive( List<Tsr<T>> inputs, int index ) {
        return derive( inputs.toArray( new Tsr[ 0 ] ), index );
    }

    // ---

    @Override
    public <T> Tsr<T> call( Tsr<T>[] inputs, int j ) {
        return (Tsr<T>) execute(inputs, j);
    }

    @Override
    public <T> Tsr<T> call( Tsr<T>... inputs ) {
        return (Tsr<T>) execute(inputs);
    }

    @Override
    public <T> Tsr<T> derive( Tsr<T>[] inputs, int d, int j ) {
        return (Tsr<T>) executeDerive( inputs, d, j );
    }

    @Override
    public <T> Tsr<T> derive( Tsr<T>[] inputs, int d ) {
        return (Tsr<T>) executeDerive( inputs, d );
    }


}
