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

    ____                       _   _          _______
   / __ \                     | | (_)        |__   __|
  | |  | |_ __   ___ _ __ __ _| |_ _  ___  _ __ | |_   _ _ __   ___
  | |  | | '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \| | | | | '_ \ / _ \
  | |__| | |_) |  __/ | | (_| | |_| | (_) | | | | | |_| | |_) |  __/
   \____/| .__/ \___|_|  \__,_|\__|_|\___/|_| |_|_|\__, | .__/ \___|
         | |                                        __/ | |
         |_|                                       |___/|_|

         The representation class for tensor operations.

------------------------------------------------------------------------------------------------------------------------

*/


package neureka.backend.api.operations;

import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.backend.api.algorithms.Algorithm;
import neureka.calculus.Function;
import neureka.backend.api.ExecutionCall;
import neureka.ndim.iterators.NDIterator;

import java.util.List;
import java.util.ServiceLoader;
import java.util.function.Consumer;

/**
 *  This interface describes an operation which
 *  ought to consist of a compositional system
 *  containing multiple algorithms which themselves
 *  ought to contain device specific implementations
 *  capable of processing ExecutionCall instances.
 *
 *  Besides the definition of the compositional system
 *  there is also the requirement for its integration into
 *  the calculus package.
 *  This means that the operation should have a function name
 *  and optionally also an operator.
 *  Alongside there must be a stringifier which ought to generate
 *  a String view as part of a Function-AST.
 */
public interface Operation
{

    @FunctionalInterface
    interface TertiaryNDIConsumer
    {
        double execute( NDIterator t0Idx, NDIterator t1Idx, NDIterator t2Idx );
    }
    @FunctionalInterface
    interface TertiaryNDXConsumer
    {
        double execute( int[] t0Idx, int[] t1Idx, int[] t2Idx );
    }
    @FunctionalInterface
    interface SecondaryNDIConsumer
    {
        double execute( NDIterator t0Idx, NDIterator t1Idx );
    }
    @FunctionalInterface
    interface SecondaryNDXConsumer
    {
        double execute( int[] t0Idx, int[] t1Idx );
    }
    @FunctionalInterface
    interface PrimaryNDIConsumer
    {
        double execute( NDIterator t0Idx );
    }
    @FunctionalInterface
    interface PrimaryNDXConsumer
    {
        double execute( int[] t0Idx );
    }

    //---

    @FunctionalInterface
    interface DefaultOperatorCreator<T>
    {
        T create( Tsr<?>[] inputs, int d );
    }
    @FunctionalInterface
    interface ScalarOperatorCreator<T>
    {
        T create( Tsr<?>[] inputs, double scalar, int d );
    }

    //==================================================================================================================

    <T extends Algorithm<T>> Algorithm<T> getAlgorithmFor( ExecutionCall<?> call );

    //==================================================================================================================

    String getFunction();

    //==================================================================================================================

    <T extends Algorithm<T>> T getAlgorithm( Class<T> type );

    <T extends Algorithm<T>> boolean supportsAlgorithm( Class<T> type );

    <T extends Algorithm<T>> Operation setAlgorithm( Class<T> type, T instance );

    //==================================================================================================================

    String stringify( String[] children );

    //==================================================================================================================

    String asDerivative( Function[] children, int d );

    //==================================================================================================================

    int getId();

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

    boolean isIndexer();

    boolean isDifferentiable();

    boolean isInline();

    boolean isPrefix();

    <T extends Algorithm<T>> boolean supports( Class<T> implementation );

    /**
     * This method mainly ought to serve as a reference- and fallback- implementation for tensor backends and also
     * as the backend for handling the calculation of scalar inputs passed to a given abstract syntax tree of
     * Function instances... <br>
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
