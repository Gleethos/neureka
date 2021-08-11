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


import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.GraphLock;
import neureka.autograd.GraphNode;
import neureka.backend.api.Operation;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.calculus.implementations.FunctionInput;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 *  Besides the {@link Tsr} class, which is the core class of Neureka, this interface and its implementations
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
    /**
     *  This static factory method will return {@link Function} instances
     *  based on a provided mathematical {@link String} expression describing the function
     *  using 'I[0]', 'I[1]', 'I[2]'... as input variables or 'I[j]' to enable input dependent indexing
     *  like for example "sum( I[j] / 2 )".
     *  The {@link Function} instances returned by this method will
     *  by default perform autograd if any involved input {@link Tsr} requires gradients (see {@link Tsr#rqsGradient()}).
     *  If one wishes to disable this behavior one might consider the use of the {@link Function#of(String, boolean)}
     *  factory method.
     *
     * @param expression The right part of a function equation where inputs are denoted by 'I[0]', 'I[1]', 'I[2]'...
     * @return A {@link Function} instance created based on the provided {@link String}, ready to receive inputs and execute on them.
     */
    static Function of( String expression ) { return of( expression, true ); }

    /**
     *  This static factory method will return {@link Function} instances
     *  based on a provided mathematical {@link String} expression describing the function
     *  using 'I[0]', 'I[1]', 'I[2]'... as input variables or 'I[j]' to enable input dependent indexing
     *  like for example "sum( I[j] / 2 )" as well as a flag determining if the resulting {@link Function}
     *  ought to be able to perform autograd or not.
     *
     * @param expression The right part of a function equation where inputs are denoted by 'I[0]', 'I[1]', 'I[2]'...
     * @param doAD A flag determining if the produced {@link Function} should be able to perform autograd (aka. auto-differentiation)
     * @return A {@link Function} instance created based on the provided {@link String}, ready to receive inputs and execute on them.
     */
    static Function of( String expression, boolean doAD ) {
        return new FunctionBuilder(Neureka.get().context()).build(expression, doAD);
    }

    /**
     *  This static nested class acts as namespace for a set of useful
     *  entry points to executing the provided parameters.
     */
    class Setup
    {
        public static <T> Tsr<T> commit( Tsr<T>[] tensors, String operation, boolean doAD )
        {
            return commit( null, tensors, new FunctionBuilder(Neureka.get().context()).build( operation, doAD ) );
        }

        public static <T> Tsr<T> commit( Tsr<T> drain, Tsr<T>[] tensors, String operation, boolean doAD )
        {
            return commit( drain, tensors, new FunctionBuilder(Neureka.get().context()).build( operation, doAD ) );
        }

        public static <T> Tsr<T> commit( Tsr<T>[] inputs, Function function )
        {
            return commit( null, inputs, function );
        }

        public static <T> Tsr<T> commit( Tsr<T> drain, Tsr<T>[] inputs, Function function )
        {
            return commit( drain, inputs, function, null );
        }

        public static <T> Tsr<T> commit(
                Tsr<?> drain, Tsr<?>[] inputs, Function function, Supplier<Tsr<Object>> activation
        ) {
            Tsr.makeFit( inputs, function.isDoingAD() ); // reshaping if needed

            GraphLock newLock = new GraphLock( function );
            for ( Tsr<?> t : inputs ) {
                if ( t.has( GraphNode.class ) ) t.find( GraphNode.class ).obtainLocking( newLock );
                else new GraphNode( function, newLock, () -> t );
            }
            Tsr<T> result;
            if ( activation == null ) result = (Tsr<T>) function.execute( inputs );
            else result = (Tsr<T>) activation.get();

            Neureka.get().context().functionCache().free( newLock );
            boolean resultIsUnique = true;
            if ( drain != null ) {
                for( Tsr<?> t : inputs ) {
                    Tsr<?> g = t.getGradient();
                    if (t == result || ( g != null && g == result ) ) {
                        resultIsUnique = false;
                        break;
                    }
                }
            }
            if ( resultIsUnique )
                return result;
            else return null;
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    Function newBuild( String expression );

    /**
     *  Only branch {@link Function}s can do autograd / 'Auto-Differentiation', meaning functions
     *  whose {@link #isFlat()} flag is set to false!
     *
     * @return The truth value determining if this {@link Function} can perform autograd/auto-differentiation on the input tensors it receives.
     */
    boolean isDoingAD();

    /**
     * @return The truth value determining if the sub-functions of this {@link Function} do not themselves reference {@link Function}s.
     */
    boolean isFlat();

    /**
     * @return The {@link Operation} implementation instance responsible for executing any inputs received by this {@link Function} or null if this {@link #isFlat()}.
     */
    Operation getOperation();

    /**
     * @param index The index which ought to match the input index of a potentially referenced {@link FunctionInput}.
     * @return The truth value determining if this {@link Function} (or any sub-functions) reference an {@link FunctionInput} with the provided index.
     */
    boolean dependsOn( int index );

    /**
     *  This method builds a new {@link Function} which is the derivative of this {@link Function} with respect to the provided input index.
     * @param index The index of the input which ought to serve as the variable which ought to be derived.
     * @return The derivative of this {@link Function}.
     */
    Function getDerivative( int index );

    List<Function> getSubFunctions();

    default List<Function> getAllFunctions() {
        return _unpack( this.getSubFunctions() );
    }

    default int numberOfArgs() {
        return (int) getAllFunctions()
                        .stream()
                        .filter( fun -> fun instanceof FunctionInput )
                        .map( fun -> (FunctionInput) fun )
                        .mapToInt( FunctionInput::index )
                        .distinct()
                        .count();
    }

    static List<Function> _unpack( List<Function> functions ) {
        List<Function> collected = new ArrayList<>();
        __unpack(functions, collected);
        return collected;
    }

    static void __unpack( List<Function> functions, List<Function> target ) {
        target.addAll(functions);
        for ( Function fun : functions ) __unpack(fun.getSubFunctions(), target);
    }


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

    Tsr<?> execute( Tsr<?>... inputs );
    Tsr<?> execute( Tsr<?>[] inputs, int j );
    Tsr<?> executeDerive( Tsr<?>[] inputs, int index, int j );
    Tsr<?> executeDerive( Tsr<?>[] inputs, int index );

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

    //------------------------------------------------------------------------------------------------------------------


    <T> Tsr<T> derive( Tsr<T>[] inputs, int index, int j );
    <T> Tsr<T> derive( Tsr<T>[] inputs, int index );

    //---

    <T> Tsr<T> derive( List<Tsr<T>> inputs, int index, int j );
    <T> Tsr<T> derive( List<Tsr<T>> inputs, int index );

    //---

    String toString();


}

 