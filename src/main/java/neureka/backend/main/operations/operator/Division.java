package neureka.backend.main.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.BiElementWise;
import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.backend.main.operations.operator.impl.CLBroadcastDivision;
import neureka.backend.main.operations.operator.impl.CLScalarBroadcastDivision;
import neureka.backend.main.operations.operator.impl.CPUBroadcastDivision;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.NDimensional;


public class Division extends AbstractOperation
{
    public Division()
    {
        super(
                new OperationBuilder()
                        .identifier(         "divide"   )
                        .operator(         "/"        )
                        .arity(            -1         )
                        .isOperator(       true       )
                        .isIndexer(        false      )
                        .isDifferentiable( true       )
                        .isInline(         false      )
        );

        //_____________________
        // DEFAULT OPERATION :

        BiElementWise biElementWise = new BiElementWise(ElemWiseUtil::forDivisionsOrModuli)
                                   .setSupplyADAgentFor( getDefaultAlgorithm() )
                                    .buildFunAlgorithm();

        setAlgorithm(
            BiElementWise.class,
            biElementWise
                .setImplementationFor(
                    CPU.class,
                    BiElementWise.implementationForCPU()
                        .with(Fun.F64F64ToF64.triple(
                            ( a, b ) -> a / b,
                            ( a, b ) -> 1 / b, // Deriving at input 0
                            ( a, b ) -> -( a / Math.pow( b, 2 ) ) // deriving input 1
                        ))
                        .with(Fun.F32F32ToF32.triple(
                            ( a, b ) -> a / b,
                            ( a, b ) -> 1f / b, // Deriving at input 0
                            ( a, b ) -> (float) -( a / Math.pow( b, 2 ) ) // deriving input 1
                        ))
                        .with(Fun.I32I32ToI32.triple(
                            ( a, b ) -> a / b,
                            ( a, b ) -> (int) Math.round(1d / b), // Deriving at input 0
                            ( a, b ) -> (int) Math.round( -a / Math.pow( b, 2 ) ) // deriving input 1
                        ))
                        .with(Fun.I64I64ToI64.triple(
                            ( a, b ) -> a / b,
                            ( a, b ) -> Math.round(1d / b), // Deriving at input 0
                            ( a, b ) -> Math.round( -a / Math.pow( b, 2 ) ) // deriving input 1
                        ))
                        .get()
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    BiElementWise.implementationForGPU( this.getIdentifier() )
                            .with( "output = input1 / input2;\n" )
                            .and(
                                    "    if ( d == 0 ) {                                   \n" +
                                            "        output = 1 / input2;                           \n" +
                                            "    } else {                                           \n" +
                                            "        output = -input2 / (float)pow(input1, 2.0f);   \n" +
                                            "    }                                                  \n"
                            )
                )
        );

        //________________
        // BROADCASTING :

        setAlgorithm(
                Broadcast.class,
                new Broadcast( ElemWiseUtil::forDivisionsOrModuli )
                .setAutogradModeFor(
                    call -> call
                            .validate().allNotNullHaveSame(NDimensional::shape)
                            .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                            .orElse(AutoDiffMode.BACKWARD_ONLY)
                )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call ) ->
                    {
                        if ( call.autogradMode().allowsForward() )
                            throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                        Function mul = Neureka.get().backend().getFunction().mul();
                        if ( ctxDerivative != null ) {
                            return ADAgent.of( ctxDerivative )
                                            .withAD( target -> mul.execute( target.error(), ctxDerivative ) );
                        }
                        int d = call.getDerivativeIndex();
                        Tsr<?> derivative = f.executeDerive( call.inputs(), d );
                        return ADAgent.of( derivative )
                                        .withAD( target -> mul.execute( target.error(), derivative ) );
                    }
                )
                .buildFunAlgorithm()
                .setImplementationFor(
                    CPU.class,
                    new CPUBroadcastDivision()
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    new CLBroadcastDivision( this.getIdentifier() )
                )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        setAlgorithm(
            Scalarization.class,
            new Scalarization()
            .setIsSuitableFor( call -> SuitabilityPredicate.BAD )
            .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
            .setDeviceExecution( (call, callback) -> ElemWiseUtil.forDivisionsOrModuli(call, callback) )
            .buildFunAlgorithm()
            .setImplementationFor(
                CPU.class,
                Scalarization.implementationForCPU()
                    .with(Fun.F64F64ToF64.triple(
                        ( a, b ) -> a / b,
                        ( a, b ) -> 1 / b, // Deriving at input 0
                        ( a, b ) -> -( a / Math.pow( b, 2 ) ) // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> a / b,
                        ( a, b ) -> 1 / b, // Deriving at input 0
                        ( a, b ) -> (float) -( a / Math.pow( b, 2 ) ) // deriving input 1
                    ))
                    .with(Fun.I32I32ToI32.triple(
                        ( a, b ) -> a / b,
                        ( a, b ) -> (int) Math.round(1d / b), // Deriving at input 0
                        ( a, b ) -> (int) Math.round( -a / Math.pow( b, 2 ) ) // deriving input 1
                    ))
                    .with(Fun.I64I64ToI64.triple(
                        ( a, b ) -> a / b,
                        ( a, b ) -> Math.round(1d / b), // Deriving at input 0
                        ( a, b ) -> Math.round( -a / Math.pow( b, 2 ) ) // deriving input 1
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                new CLScalarBroadcastDivision( this.getIdentifier() )
            )
        );
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return _asDerivative( children, derivationIndex, children.length - 1 );
    }

    private String _asDerivative( Function[] children, int d, int index ) {
        if ( d >= 0 ) {
            if ( index <= 0 ) return children[ 0 ].getDerivative( d ).toString();
            else {
                String first = ( children[ index - 1 ].dependsOn( d ) )
                        ? "(" + _asDerivative( children, d, index - 1 )+ " / " + children[ index ]  + " )"
                        : "";

                if ( !children[ index ].dependsOn(d) ) return first;
                String s = children[ index - 1 ].toString();
                if ( s.equals("0.0") ) return first;

                return first +
                        " - ((" + // The second expression is the inner derivative (current index)! (inner times outer...)
                            s + " * " + children[ index ].getDerivative(d) +
                        ") / ( "
                            + children[ index ] + "**2 " +
                        ") )";
            }
        } else {
            if ( index <= 0 ) return children[ 0 ].toString();
            else
                return _asDerivative( children, -1, index - 1 ) + " / " + children[ index ].toString();
        }
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result /= current;
            }
            return result;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call( inputs, j );
            ud = src[ 0 ].derive( inputs, d, j );
            for ( int i = 0; i < src.length - 1; i++ ) {
                v = src[ i + 1 ].call( inputs, j );
                vd = src[ i + 1 ].derive( inputs, d, j );
                ud = (ud * v - u * vd) / Math.pow(v, 2);
                u /= v;
            }
            return ud;
        }
    }

    
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result /= current;
            }
            return result;
        } else {
            double derivative;
            double tempVar = src[ 0 ].call( inputs );
            derivative = src[ 0 ].derive( inputs, d );

            for ( int i = 0; i < src.length - 1; i++ ) {
                double u, ud, v, vd;
                v = src[ i + 1 ].call( inputs );
                vd = src[ i + 1 ].derive( inputs, d );
                u = tempVar;
                ud = derivative;
                derivative = ( ud * v - u * vd ) / Math.pow(v, 2);
                tempVar /= v;
            }
            return derivative;
        }
    }




}
