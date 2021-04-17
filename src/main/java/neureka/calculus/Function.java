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

   ______                _   _
  |  ____|              | | (_)
  | |__ _   _ _ __   ___| |_ _  ___  _ __
  |  __| | | | '_ \ / __| __| |/ _ \| '_ \
  | |  | |_| | | | | (__| |_| | (_) | | | |
  |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|

    Function instances represent an abstract syntax
    tree which performs operations on tensors or arrays of primitive scalars.

*/

package neureka.calculus;

import neureka.Tsr;
import neureka.autograd.GraphLock;
import neureka.autograd.GraphNode;
import neureka.backend.api.Operation;
import neureka.calculus.assembly.FunctionBuilder;

import java.util.List;
import java.util.function.Supplier;

/**
 *  Besides the Tsr class, which is the core class of Neureka, this interface and its implementations
 *  represents the second most important feature of this library.
 *  Instances of {@link Function} implementations form an abstract syntax tree which is being built
 *  from a provided expression {@link String} containing function syntax.
 *
 *  Just like functions in the mathematical sense, implementations of this
 *  interface receive a fixed number of inputs.
 *  Within the expression String needed for instantiation, these inputs are
 *  recognized by 'I[j]', 'Ij' or 'ij', where j is the input index.
 *  Functions accept arrays as their inputs,
 *  which is why variables must be targeted in such a way.
 */
public interface Function
{
    // Global context and cache:
    Cache CACHE = Cache.instance();

    Function DIMTRIM = create("dimtrim(I[ 0 ])");

    Function IDY = create("I[ 0 ] <- I[ 1 ]");

    Function X = create("I[ 0 ] x I[ 1 ]");
    Function PLUS = create("(I[ 0 ] + I[ 1 ])");
    Function PLUS_ASSIGN = create("I[ 0 ] <- (I[ 0 ] + I[ 1 ])");
    Function MINUS = create("(I[ 0 ] - I[ 1 ])");
    Function MINUS_ASSIGN = create("I[ 0 ] <- (I[ 0 ]-I[ 1 ])");
    Function DIV = create("(I[ 0 ]/I[ 1 ])");
    Function DIV_ASSIGN = create("I[ 0 ] <- (I[ 0 ] / I[ 1 ])");
    Function POW = create("(I[ 0 ] ^ I[ 1 ])");
    Function POW_ASSIGN = create("I[ 0 ] <- (I[ 0 ] ^ I[ 1 ])");
    Function MUL = create("I[ 0 ] * I[ 1 ]");
    Function MUL_ASSIGN = create("I[ 0 ] <- (I[ 0 ] * I[ 1 ])");
    Function MOD = create("(I[ 0 ] % I[ 1 ])");
    Function MOD_ASSIGN = create("I[ 0 ] <- (I[ 0 ] % I[ 1 ])");
    Function NEG = create("(-1 * I[ 0 ])");

    /**
     *  This static nested class acts as namespace for pre-instantiated
     *  Function instances which are configured to not track their computational history.
     *  This means that no computation graph will be built by these instances.
     *  ( Computation graphs in Neureka are made of instances of the "GraphNode" class... )
     */
    class Detached
    {
        public static Function IDY = create("I[ 0 ]<-I[ 1 ]", false);

        public static Function X = create("I[ 0 ]xI[ 1 ]", false);
        public static Function PLUS = create("(I[ 0 ]+I[ 1 ])", false);
        public static Function PLUS_ASSIGN = create("I[ 0 ]<-(I[ 0 ]+I[ 1 ])", false);
        public static Function MINUS = create("(I[ 0 ]-I[ 1 ])", false);
        public static Function MINUS_ASSIGN = create("I[ 0 ]<-(I[ 0 ]-I[ 1 ])", false);
        public static Function DIV = create("(I[ 0 ]/I[ 1 ])", false);
        public static Function DIV_ASSIGN = create("I[ 0 ]<-(I[ 0 ]/I[ 1 ])", false);
        public static Function POW = create("(I[ 0 ]^I[ 1 ])", false);
        public static Function POW_ASSIGN = create("I[ 0 ]<-(I[ 0 ]^I[ 1 ])", false);
        public static Function MUL = create("I[ 0 ]*I[ 1 ]", false);
        public static Function MUL_ASSIGN = create("I[ 0 ]<-(I[ 0 ]*I[ 1 ])", false);
        public static Function ADD = create("I[ 0 ]+I[ 1 ]", false);
        public static Function ADD_ASSIGN = create("I[ 0 ]<-(I[ 0 ]+I[ 1 ])", false);
        public static Function MOD = create("(I[ 0 ]%I[ 1 ])", false);
        public static Function MOD_ASSIGN = create("I[ 0 ]<-(I[ 0 ]%I[ 1 ])", false);
        public static Function NEG = create("(-1*I[ 0 ])", false);
    }

    static Function create( String expression ) {
        return create( expression, true );
    }

    static Function create( String expression, boolean doAD ) {
        return FunctionBuilder.build(expression, doAD);
    }

    /**
     *  This static nested class acts as namespace for a set of useful
     *  entry points to
     */
    class Setup
    {
        public static <T> Tsr<T> commit( Tsr<T>[] tensors, String operation, boolean doAD )
        {
            return commit( null, tensors, FunctionBuilder.build( operation, doAD ) );
        }

        public static <T> Tsr<T> commit( Tsr<T> drain, Tsr<T>[] tensors, String operation, boolean doAD )
        {
            return commit( drain, tensors, FunctionBuilder.build( operation, doAD ) );
        }

        public static <T> Tsr<T> commit( Tsr<T>[] inputs, Function function )
        {
            return commit( null, inputs, function );
        }

        public static <T> Tsr<T> commit( Tsr<T> drain, Tsr<T>[] inputs, Function function )
        {
            return commit( drain, inputs, function, null );
        }

        public static <T> Tsr<T> commit( Tsr<T> drain, Tsr<T>[] inputs, Function function, Supplier<Tsr<T>> activation )
        {
            Tsr.makeFit( inputs, function.isDoingAD() ); // reshaping if needed

            GraphLock newLock = new GraphLock( function, inputs );
            for ( Tsr<T> t : inputs ) {
                if ( t.has( GraphNode.class ) ) t.find( GraphNode.class ).obtainLocking( newLock );
                else new GraphNode( function, newLock, () -> t );
            }
            Tsr<T> result;
            if ( activation == null ) result = function.call( inputs );
            else result = activation.get();

            Function.CACHE.free( newLock );
            boolean resultIsUnique = true;
            if ( drain != null ) {
                for( Tsr<T> t : inputs ) {
                    Tsr<T> g = t.getGradient();
                    if (t == result || ( g != null && g == result ) ) {
                        resultIsUnique = false;
                        break;
                    }
                }
            }
            if ( resultIsUnique ) return result;
            else return null;
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    Function newBuild( String expression );

    boolean isDoingAD();//Note: only branch nodes can 'do Auto-Differentiation'

    boolean isFlat();

    Operation getOperation();

    boolean dependsOn( int index );

    Function getDerivative( int index );

    //------------------------------------------------------------------------------------------------------------------

    double call( double input );
    double invoke( double input );

    //------------------------------------------------------------------------------------------------------------------

    double call( double[] inputs, int j );
    double invoke( double[] inputs, int j );


    double call( double... inputs );
    double invoke( double... inputs );


    double derive( double[] inputs, int index, int j );
    double derive( double[] inputs, int index );

    //------------------------------------------------------------------------------------------------------------------

    <T> Tsr<T> call( Tsr<T> input );
    <T> Tsr<T> invoke( Tsr<T> input );

    <T> Tsr<T> call( List<Tsr<T>> input );
    <T> Tsr<T> invoke( List<Tsr<T>> input );

    //------------------------------------------------------------------------------------------------------------------

    <T> Tsr<T> call( Tsr<T>[] inputs, int j );
    <T> Tsr<T> invoke( Tsr<T>[] inputs, int j );

    <T> Tsr<T> call( Tsr<T>... inputs );
    <T> Tsr<T> invoke( Tsr<T>... inputs );


    <T> Tsr<T> derive( Tsr<T>[] inputs, int index, int j );
    <T> Tsr<T> derive( Tsr<T>[] inputs, int index );

    //---

    <T> Tsr<T> derive( List<Tsr<T>> inputs, int index, int j );
    <T> Tsr<T> derive( List<Tsr<T>> inputs, int index );

    //---

    String toString();


}

 