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
import neureka.backend.api.Call;
import neureka.backend.api.Operation;
import neureka.backend.standard.memory.MemUtil;
import neureka.backend.standard.memory.MemValidator;
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.calculus.implementations.FunctionInput;
import neureka.devices.Device;

import java.util.ArrayList;
import java.util.List;

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
        return new FunctionBuilder( Neureka.get().backend() ).build( expression, doAD );
    }

    //------------------------------------------------------------------------------------------------------------------

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
        List<Function> allFuns = new ArrayList<>();
        allFuns.add(this);
        for ( Function fun : this.getSubFunctions() ) {
            allFuns.addAll(fun.getAllFunctions());
        }
        return allFuns;
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

    //------------------------------------------------------------------------------------------------------------------

    default double call( double input )   { return call( new double[]{input} ); }
    default double invoke( double input ) { return call( input               ); }

    //------------------------------------------------------------------------------------------------------------------

    double call( double[] inputs, int j );
    default double invoke( double[] inputs, int j ) { return call( inputs, j ); }


    default double call( double... inputs )   { return call( inputs, -1 ); }
    default double invoke( double... inputs ) { return call( inputs        ); }


    double derive( double[] inputs, int index, int j );
    double derive( double[] inputs, int index        );

    //------------------------------------------------------------------------------------------------------------------

    default <T, D extends Device<T>> Tsr<T> call( Call.Builder<T, D> call ) {
        Tsr<T> result = (Tsr<T>) execute( call.get() );
        result.getUnsafe().setIsIntermediate(false);
        return result;
    }

    default <T, D extends Device<T>> Tsr<T> invoke( Call.Builder<T, D> call ) {
        Tsr<T> result = (Tsr<T>) execute( call.get() );
        result.getUnsafe().setIsIntermediate(false);
        return result;
    }

    /**
     *  <b>Warning: Tensors returned by this method are eligible for deletion when consumed by other function.</b>
     */
    default Tsr<?> execute( Call<?> call ) {
        /*
            In order to dispatch the user call to the backend we need to
            convert the call to a format compatible with the backend!
            For that we simply need a list of arguments and tensors.
         */
        List<Arg> args = call.allMetaArgs();
        if ( call.getDevice() != null ) args.add(Arg.TargetDevice.of((Device<?>) call.getDevice()));
        Arg<?>[] argArray = new Arg[args.size()];
        for ( int i = 0; i < argArray.length; i++ ) argArray[i] = args.get(i);
        return callWith(argArray).execute(call.getTensors());
    }

    /**
     *  An API for calling a {@link Function} after having specified
     *  a set of {@link Arg} instances through the {@link #callWith(Args)}
     *  method.
     */
    interface CallOptions {
        <T> Tsr<T> call( Tsr<T>... tensors );
        <T> Tsr<T> invoke( Tsr<T>... tensors );
        /**
         *  <b>Warning: Tensors returned by this method are eligible for deletion when consumed by other function.</b>
         *
         * @param tensors The tensors which should be sent through this function.
         * @return The result from the execution of the provided tensors.
         */
        Tsr<?> execute( Tsr<?>... tensors );
    }

    /**
     * @param arguments A set of arguments you want to supply to this function for further
     *                  control over the execution.
     *
     * @return A simple API for passing the {@link Tsr} arguments and calling this {@link Function}.
     */
    default CallOptions callWith( Arg<?>... arguments ) { return callWith( Args.of( arguments ) ); }

    /**
     * @param arguments A set of arguments you want to supply to this function for further
     *                  control over the execution.
     *
     * @return A simple API for passing the {@link Tsr} arguments and calling this {@link Function}.
     */
    default CallOptions callWith( Args arguments ) {
       return new CallOptions() {
           @SafeVarargs @Override public final <T> Tsr<T> call(    Tsr<T>... tensors ) { return (Tsr<T>) this.execute( tensors ); }
           @SafeVarargs @Override public final <T> Tsr<T> invoke(  Tsr<T>... tensors ) { return (Tsr<T>) this.execute( tensors ); }
           @Override public Tsr<?> execute( Tsr<?>... tensors )
           {
               MemValidator validation = MemValidator.forInputs( tensors, ()->Function.this.execute( arguments, tensors ) );
               if ( validation.isWronglyIntermediate() ) {
                   throw new IllegalStateException(
                           "Output of function '" + Function.this + "' " +
                           (Function.this.getOperation() != null ? "(" + Function.this.getOperation().getFunction() + ") " : "") +
                           "is marked as intermediate result, despite the fact " +
                           "that it is a member of the input array. " +
                           "Tensors instantiated by library users instead of operations in the backend are not supposed to be flagged " +
                           "as 'intermediate', because they are not eligible for deletion!"
                   );
               }
               if ( validation.isWronglyNonIntermediate() ) {
                   throw new IllegalStateException(
                           "Output of function '" + Function.this + "' " +
                           (Function.this.getOperation() != null ? "(" + Function.this.getOperation().getFunction() + ") " : "") +
                           "is neither marked as intermediate result nor a member of the input array. " +
                           "Tensors instantiated by operations in the backend are expected to be flagged " +
                           "as 'intermediate' in order to be eligible for deletion!"
                   );
               }
               MemUtil.autoDelete( tensors );
               return validation.getResult();
           }
       };
    }

    default <T> Tsr<T> call( Args arguments, Tsr<T>... tensors ) {
        Tsr<T> result = callWith( arguments ).call( tensors );
        return result.getUnsafe().setIsIntermediate(false);
    }

    default <T> Tsr<T> invoke( Args arguments, Tsr<T>... tensors ) {
        Tsr<T> result = callWith( arguments ).invoke( tensors );
        return result.getUnsafe().setIsIntermediate(false);
    }

    /**
     *  <b>Warning: Tensors returned by this method are eligible for deletion when consumed by other function.</b>
     */
    Tsr<?> execute( Args arguments, Tsr<?>... tensors );

    /**
     *  <b>Warning: Tensors returned by this method are eligible for deletion when consumed by other function.</b>
     */
    default Tsr<?> execute( Tsr<?>... inputs ) { return execute( inputs, -1 ); }

    /**
     *  <b>Warning: Tensors returned by this method are eligible for deletion when consumed by other function.</b>
     */
    default Tsr<?> execute( Tsr<?>[] inputs, int j ) { return callWith(Args.of(Arg.DerivIdx.of(-1), Arg.VarIdx.of(j))).execute(inputs); }
    /**
     *  <b>Warning: Tensors returned by this method are eligible for deletion when consumed by other function.</b>
     */
    default Tsr<?> executeDerive( Tsr<?>[] inputs, int index, int j ) { return callWith(Args.of(Arg.DerivIdx.of(index), Arg.VarIdx.of(j))).execute(inputs); }
    /**
     *  <b>Warning: Tensors returned by this method are eligible for deletion when consumed by other function.</b>
     */
    default Tsr<?> executeDerive( Tsr<?>[] inputs, int index ) { return executeDerive( inputs, index, -1 ); }

    //------------------------------------------------------------------------------------------------------------------

    default <T> Tsr<T> call( Tsr<T> input )   { return call( new Tsr[]{ input } ); }
    default <T> Tsr<T> invoke( Tsr<T> input ) { return call( input );            }

    default <T> Tsr<T> call( List<Tsr<T>> input )   { return call( input.toArray(new Tsr[ 0 ]) ); }
    default <T> Tsr<T> invoke( List<Tsr<T>> input ) { return call( input );                       }

    //------------------------------------------------------------------------------------------------------------------

    default <T> Tsr<T> call( Tsr<T>[] inputs, int j )   {
        Tsr<T> result = (Tsr<T>) execute( inputs, j );
        return result.getUnsafe().setIsIntermediate(false);
    }

    default <T> Tsr<T> invoke( Tsr<T>[] inputs, int j ) { return call( inputs, j );             }

    default <T> Tsr<T> call( Tsr<T>... inputs )   {
        Tsr<T> result = (Tsr<T>) execute( inputs );
        return result.getUnsafe().setIsIntermediate(false);
    }

    default <T> Tsr<T> invoke( Tsr<T>... inputs ) { return call( inputs );             }

    //------------------------------------------------------------------------------------------------------------------

    default <T> Tsr<T> derive( Tsr<T>[] inputs, int d, int j ) {
        Tsr<T> result = (Tsr<T>) executeDerive( inputs, d, j );
        return result.getUnsafe().setIsIntermediate(false);
    }

    default <T> Tsr<T> derive( Tsr<T>[] inputs, int d ) {
        Tsr<T> result = (Tsr<T>) executeDerive( inputs, d );
        return result.getUnsafe().setIsIntermediate(false);
    }

    //---

    default <T> Tsr<T> derive( List<Tsr<T>> inputs, int index, int j ) { return derive( inputs.toArray( new Tsr[ 0 ] ), index, j ); }
    default <T> Tsr<T> derive( List<Tsr<T>> inputs, int index )        { return derive( inputs.toArray( new Tsr[ 0 ] ), index ); }

    //---

    String toString();


}

 