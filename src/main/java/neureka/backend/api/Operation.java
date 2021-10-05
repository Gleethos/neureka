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

    ____                       _   _
   / __ \                     | | (_)
  | |  | |_ __   ___ _ __ __ _| |_ _  ___  _ __
  | |  | | '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \
  | |__| | |_) |  __/ | | (_| | |_| | (_) | | | |
   \____/| .__/ \___|_|  \__,_|\__|_|\___/|_| |_|
         | |
         |_|

    The representation for operations on tensors.

------------------------------------------------------------------------------------------------------------------------
*/


package neureka.backend.api;

import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.backend.api.operations.OperationBuilder;
import neureka.calculus.Function;
import neureka.ndim.iterators.NDIterator;

/**
 *  This interface is part of the backend API and it embodies the top layer of the 3 tier backend architecture.
 *  It represents broad and high level requests for execution which might be executed differently depending
 *  on the provided {@link ExecutionCall} arguments.
 *  An {@link Operation} implementation ought to consist of a component system
 *  containing multiple {@link Algorithm} instances, which themselves ought to contain device specific implementations
 *  capable of processing {@link ExecutionCall} instances, or rather their state. <br><br>
 *
 *  Besides the component system, there is also the definition for how its supposed to integrate into
 *  the {@link neureka.calculus} package in order to serve as part of an {@link Function} AST.
 *  This means that the operation should have a function name
 *  and optionally also an operator in the form of {@link String} instances.
 *  Alongside there must be an implementation of the {@link Operation#stringify(String[])} method,
 *  which ought to generate a String view as part of a {@link Function}-AST.
 */
public interface Operation
{
    static OperationBuilder builder() { return new OperationBuilder(); }

    @FunctionalInterface
    interface TertiaryNDIConsumer { double execute( NDIterator t0Idx, NDIterator t1Idx, NDIterator t2Idx ); }
    @FunctionalInterface
    interface TertiaryNDAConsumer { double execute( int[] t0Idx, int[] t1Idx, int[] t2Idx ); }
    @FunctionalInterface
    interface SecondaryNDIConsumer { double execute( NDIterator t0Idx, NDIterator t1Idx ); }
    @FunctionalInterface
    interface SecondaryNDAConsumer { double execute( int[] t0Idx, int[] t1Idx ); }
    @FunctionalInterface
    interface PrimaryNDIConsumer { double execute( NDIterator t0Idx ); }
    @FunctionalInterface
    interface PrimaryNDAConsumer { double execute( int[] t0Idx ); }

    //---

    @FunctionalInterface
    interface DefaultOperatorCreator<T> {  T create( Tsr<?>[] inputs, int d ); }
    @FunctionalInterface
    interface ScalarOperatorCreator<T>
    {  T create( Tsr<?>[] inputs, double scalar, int d ); }

    //==================================================================================================================

    Algorithm<?>[] getAllAlgorithms();

    /**
     *  Alongside a component system made up of {@link Algorithm} instances, implementations
     *  of this interface also ought to express a routing mechanism which finds the best {@link Algorithm}
     *  for a given {@link ExecutionCall} instance.
     *  This method signature describes this requirement.
     *
     * @param call The {@link ExecutionCall} instance which needs the best {@link Algorithm} for execution.
     * @param <T> The type parameter describing the concrete type of the {@link Algorithm} implementation.
     * @return The chosen {@link Algorithm} which ought to be fir for execution the provided call.
     */
    <T extends Algorithm<T>> Algorithm<T> getAlgorithmFor( ExecutionCall<?> call );

    //==================================================================================================================

    /**
     *  Concrete {@link Operation} types ought to be representable by a function name.
     *  The following ensures that this contract is met when overriding the method.
     *
     * @return the function name which serves as identifier when parsing {@link Function} instances.
     */
    String getFunction();

    //==================================================================================================================

    /**
     *  {@link Operation} implementations embody a component system hosting unique {@link Algorithm} instances.
     *  For a given class implementing the {@link Algorithm} class, there can only be a single
     *  instance of it referenced (aka supported) by a given {@link Operation} instance.
     *  This method enables the registration of {@link Algorithm} types in the component system of this {@link Operation}.
     *
     * @param type The class of the type which implements {@link Algorithm} as key for the provided instance.
     * @param instance The instance of the provided type class which ought to be referenced (supported) by this {@link Operation}.
     * @param <T> The type parameter of the {@link Algorithm} type class.
     * @return This very {@link Operation} instance to enable method chaining on it.
     */
    <T extends Algorithm<T>> Operation setAlgorithm( Class<T> type, T instance );

    default <T extends Algorithm<T>> Operation setAlgorithm( T instance ) {
        return setAlgorithm( (Class<T>) instance.getClass(), instance );
    }

    /**
     *  {@link Operation} implementations embody a component system hosting unique {@link Algorithm} instances.
     *  For a given class implementing the {@link Algorithm} class, there can only be a single
     *  instance of it referenced (aka supported) by a given {@link Operation} instance.
     *  This method ensures this in terms of read access by returning only a single instance or null
     *  based on the provided class instance whose type extends the {@link Algorithm} interface.
     *
     * @param type The class of the type which implements {@link Algorithm} as a key to get an existing instance.
     * @param <T> The type parameter of the {@link Algorithm} type class.
     * @return The instance of the specified type if any exists within this {@link Operation}.
     */
    <T extends Algorithm<T>> T getAlgorithm( Class<T> type );

    /**
     *  This method checks if this {@link Operation} contains an instance of the
     *  {@link Algorithm} implementation specified via its type class.
     *
     * @param type The class of the type which implements {@link Algorithm}.
     * @param <T> The type parameter of the {@link Algorithm} type class.
     * @return The truth value determining if this {@link Operation} contains an instance of the specified {@link Algorithm} type.
     */
    <T extends Algorithm<T>> boolean supportsAlgorithm( Class<T> type );

    //==================================================================================================================

    String stringify( String[] children );

    //==================================================================================================================

    /**
     *  {@link Operation} implementations and {@link Function} implementations are in a tight relationship
     *  where the {@link Function} describes an abstract syntax tree based on the syntactic information provided
     *  by the {@link Operation} (through methods like {@link Operation#getOperator()} or {@link Operation#getFunction()}).
     *  One important feature of the {@link Function} is the ability to create
     *  derivatives by calling the {@link Function#getDerivative(int)} method.
     *  Implementations of this {@link Function} method ought to call the method defined below in order to
     *  form the derivation based on the child nodes of the abstract syntax tree of the given {@link Function} node.
     *
     * @param children The child nodes of a AST node referencing this operation.
     * @param derivationIndex The index of the input node which ought to be derived.
     * @return The derivative as a {@link String} which should be parsable into yet another AST.
     */
    String asDerivative( Function[] children, int derivationIndex );

    //==================================================================================================================

    String getOperator();

    /**
     * Arity is the number of arguments or operands
     * that this function or operation takes.
     */
    int getArity();

    /**
     *  An operator is an alternative to a function like "sum()" or "prod()". <br>
     *  Examples would be "+, -, * ..."!
     *
     * @return If this operation can be represented as operator like "+, -, * ..."!
     */
    boolean isOperator();

    /**
     *  This boolean property tell the {@link Function} implementations that this {@link Operation}
     *  ought to be viewed as something to be indexed.
     *  The {@link Function} will use this information to iterate over all the provided inputs and
     *  then execute the function wile also passing the index to the function AST.
     *  The resulting array will then be available to this {@link Operation} as argument list.
     *  This feature works alongside the {@link Function} implementation found in
     *  {@link neureka.calculus.implementations.FunctionVariable}, which represents an input indexed
     *  by the identifier 'j'!
     *
     * @return If this operation is an indexer.
     */
    boolean isIndexer();

    /**
     *  This has currently no use!
     */
    @Deprecated
    boolean isDifferentiable();

    /**
     *  This flag indicates that the implementation of this {@link Operation}
     *  performs an operation which modifies the inputs to that operation.
     *  An example of this would be an assignment operation which copies the contents of one nd-array / tensor
     *  into another tensor. This second tensor will then have changed its state.
     *  This can be dangerous when auto-differentiation is involved.
     *
     * @return The truth value determining if this {@link Operation} changes the contents of inputs.
     */
    boolean isInline();

    <T extends Algorithm<T>> boolean supports( Class<T> implementation );

    /**
     * This method mainly ought to serve as a reference- and fallback- implementation for tensor backends and also
     * as the backend for handling the calculation of scalar inputs passed to a given abstract syntax tree of
     * {@link Function} instances... <br>
     * ( (almost) every Function instance contains an OperationType reference to which it passes scalar executions... )
     * <br><br>
     * This is also the reason why the last parameter of this method is a list of Function objects :
     * The list stores the child nodes of the Function node that is currently being processed.
     * Therefore when implementing this method one should first call the child nodes in
     * order to get the "real inputs" of this current node.
     * <br><br>
     * One might ask : Why does that not happen automatically?
     * The answer is to that question lies in the other parameters of this method.
     * Specifically the parameter "d" !
     * This argument determines if the derivative ought to be calculated and
     * also which value is being targeted within the input array.
     * Depending on this variable and also the nature of the operation,
     * the execution calls to the child nodes of this node change considerably!
     * <br><br>
     *
     * @param inputs An array of scalar input variables.
     * @param j The index variable for indexed execution on the input array. (-1 if no indexing should occur)
     * @param d The index of the variable of which a derivative ought to be calculated.
     * @param src The child nodes of the Function node to which this very OperationType belongs.
     * @return The result of the calculation.
     */
    double calculate( double[] inputs, int j, int d, Function[] src );


    /**
     *  This static utility class contains simple methods used for creating slices of plain old
     *  arrays of tensor objects...
     *  These slices may be used for many reasons, however mainly when iterating over
     *  inputs to a Function recursively in order to execute them pairwise for example...
     */
    class Utility
    {
        public static Tsr<?>[] subset( Tsr<?>[] tsrs, int padding, int index, int offset ) {
            if ( offset < 0 ) {
                index += offset;
                offset *= -1;
            }
            Tsr<?>[] newTsrs = new Tsr[ offset + padding ];
            System.arraycopy( tsrs, index, newTsrs, padding, offset );
            return newTsrs;
        }

        public static Tsr<?>[] without( Tsr<?>[] tsrs, int index ) {
            Tsr<?>[] newTsrs = new Tsr[ tsrs.length - 1 ];
            for ( int i = 0; i < newTsrs.length; i++ ) newTsrs[ i ] = tsrs[ i + ( ( i < index ) ? 0 : 1 ) ];
            return newTsrs;
        }

        public static Tsr<?>[] offsetted( Tsr<?>[] tsrs, int offset ) {
            Tsr<?>[] newTsrs = new Tsr[ tsrs.length - offset ];
            newTsrs[ 0 ] = Tsr.Create.newTsrLike( tsrs[ 1 ] );
            if ( !tsrs[ 1 ].has( GraphNode.class ) && tsrs[ 1 ] != tsrs[ 0 ] ) {//Deleting intermediate results!
                tsrs[ 1 ].delete();
                tsrs[ 1 ] = null;
            }
            if (!tsrs[ 2 ].has( GraphNode.class ) && tsrs[ 2 ] != tsrs[ 0 ]) {//Deleting intermediate results!
                tsrs[ 2 ].delete();
                tsrs[ 2 ] = null;
            }
            System.arraycopy( tsrs, 1 + offset, newTsrs, 1, tsrs.length - 1 - offset );
            newTsrs[ 1 ] = tsrs[ 0 ];
            return newTsrs;
        }

    }

}
