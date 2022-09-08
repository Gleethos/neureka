package neureka.backend.main.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.BiElementWise;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.implementations.CPUImplementation;
import neureka.backend.main.memory.MemUtil;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.backend.main.operations.operator.impl.CLBroadcastMultiplication;
import neureka.backend.main.operations.operator.impl.CPUBiElementWiseMultiplication;
import neureka.backend.main.operations.operator.impl.CPUBroadcastMultiplication;
import neureka.backend.main.operations.operator.impl.CPUScalarBroadcastMultiplication;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

import java.util.Arrays;
import java.util.stream.Collectors;


public class Multiplication extends AbstractOperation
{
    public Multiplication()
    {
        super(
                new OperationBuilder()
                        .identifier(    "multiply"    )
                        .operator(         "*"        )
                        .arity(            -1         )
                        .isOperator(       true       )
                        .isIndexer(        false      )
                        .isDifferentiable( true       )
                        .isInline(         false      )
        );

        //_____________________
        // DEFAULT OPERATION :

        setAlgorithm(
            BiElementWise.class,
            new BiElementWise(ElemWiseUtil::forMultiplications)
            .setSupplyADAgentFor( getDefaultAlgorithm() )
            .buildFunAlgorithm()
            .setImplementationFor( CPU.class, new CPUBiElementWiseMultiplication() )
            .setImplementationFor(
                OpenCLDevice.class,
                BiElementWise.implementationForGPU( this.getIdentifier() )
                        .with( "output = input1 * input2;\n" )
                        .and( "if ( d == 0 ) {output = input2;}else{output = input1;}\n" )
            )
        );


        //________________
        // BROADCASTING :;

        setAlgorithm(
            Broadcast.class,
            new Broadcast( ElemWiseUtil::forMultiplications )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
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
                    Tsr<?> derivative = MemUtil.keep( call.inputs(), () -> f.executeDerive( call.inputs(), d ) );
                    return ADAgent.of( derivative )
                            .withAD( target -> mul.execute( target.error(), derivative ) );
                }
            )
            .buildFunAlgorithm()
            .setImplementationFor(
                CPU.class,
                new CPUBroadcastMultiplication()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                new CLBroadcastMultiplication( this.getIdentifier() )
            )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        setAlgorithm(
            Scalarization.class,
            new Scalarization()
            .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
            .setDeviceExecution( (call, callback) -> ElemWiseUtil.forMultiplications(call, callback) )
            .buildFunAlgorithm()
            .setImplementationFor( CPU.class, new CPUScalarBroadcastMultiplication() )
            .setImplementationFor(
                OpenCLDevice.class,
                CLImplementation
                    .compiler()
                    .arity( 3 )
                    .kernelSource( Scalarization.getKernelSource() )
                    .activationSource( "output = input1 * value;\n" )
                    .differentiationSource( "if ( d == 0 ) {output = value;}else{output = input1;}\n" )
                    .kernelPostfix( this.getIdentifier() )
                    .execution(
                        call -> {
                            if ( call.getDerivativeIndex() == 0 )
                                return call.input( 2 ).shallowCopy().getUnsafe().setIsIntermediate( true );
                            else if ( call.getDerivativeIndex() == 1 )
                                return call.input( 1 ).shallowCopy().getUnsafe().setIsIntermediate( true );
                            else {
                                int offset = (call.input(Number.class, 2).isVirtual() || call.input(Number.class, 2).size() == 1) ? 1 : 0;
                                int gwz = call.input(Number.class, 0).size();
                                call.getDevice()
                                    .getKernel(call)
                                    .passAllOf(call.input(Number.class, 0))
                                    .passAllOf(call.input(Number.class, 0 + offset))
                                    .pass( call.input( Number.class, 1 + offset ).at( 0 ).get().floatValue() )
                                    .pass(call.input(Number.class, 0).rank())
                                    .pass(call.getValOf(Arg.DerivIdx.class))
                                    .call(gwz);
                            }
                            return call.input( 0 );
                        }
                    )
                    .build()
            )
        );

    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return Arrays.stream( children )
                .filter( child -> child.dependsOn(derivationIndex) )
                .map( child -> {
                            String derivative = child.getDerivative(derivationIndex).toString();
                            return ( (derivative.equals("1.0") ) ? "" : " * " ) +
                                    Arrays.stream( children )
                                        .filter( inner -> inner != child )
                                        .map( Object::toString )
                                        .collect( Collectors.joining( " * " ) );
                        }
                )
                .map( Object::toString )
                .collect( Collectors.joining( " + " ) );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result *= current;
            }
            return result;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call( inputs, j );
            ud = src[ 0 ].derive( inputs, d, j );

            for ( int ji = 1; ji < src.length; ji++ ) {
                v = src[ ji ].call( inputs, j );
                vd = src[ ji ].derive( inputs, d, j );
                ud = u * vd + v * ud;
                u *= v;
            }
            return ud;
        }
    }

    
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result *= current;
            }
            return result;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call( inputs );
            ud = src[ 0 ].derive( inputs, d );
            for ( int j = 1; j < src.length; j++ ) {
                v = src[ j ].call( inputs );
                vd = src[ j ].derive( inputs, d );

                ud = u * vd + v * ud;
                u *= v; // ...this step can be avoided (TODO optimize)
            }
            return ud;
        }
    }




}
