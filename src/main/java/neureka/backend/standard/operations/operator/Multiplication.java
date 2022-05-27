package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.fun.AutoDiffMode;
import neureka.backend.api.algorithms.fun.Result;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.algorithms.Operator;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.backend.standard.memory.MemUtil;
import neureka.backend.standard.operations.JunctionUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.internal.CalcUtil;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

import java.util.Arrays;
import java.util.stream.Collectors;


public class Multiplication extends AbstractOperation
{
    public Multiplication()
    {
        super(
                new OperationBuilder()
                        .setIdentifier(         "multiply"    )
                        .setOperator(         "*"        )
                        .setArity(            -1         )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( true       )
                        .setIsInline(         false      )
        );

        //_____________________
        // DEFAULT OPERATION :

        Operator operator = new Operator(JunctionUtil::forMultiplications)
                                   .setSupplyADAgentFor( getDefaultAlgorithm() )
                                   .buildFunAlgorithm();

        setAlgorithm(
            Operator.class,
            operator.setImplementationFor(
                CPU.class,
                Operator.implementationForCPU()
                    .with(Fun.F64F64ToF64.triple(
                        ( a, b ) -> a * b,
                        ( a, b ) -> b, // Deriving at input 0
                        ( a, b ) -> a  // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> a * b,
                        ( a, b ) -> b, // Deriving at input 0
                        ( a, b ) -> a  // deriving input 1
                    ))
                    .with(Fun.I32I32ToI32.triple(
                            ( a, b ) -> a * b,
                            ( a, b ) -> b, // Deriving at input 0
                            ( a, b ) -> a  // deriving input 1
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                Operator.implementationForGPU( this.getIdentifier() )
                        .with( "output = input1 * input2;\n" )
                        .and( "if ( d == 0 ) {output = input2;}else{output = input1;}\n" )
            )
        );


        //________________
        // BROADCASTING :

        Broadcast broadcast = new Broadcast( JunctionUtil::forMultiplications )
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
                .buildFunAlgorithm();

        setAlgorithm(
            Broadcast.class,
            broadcast
                .setImplementationFor(
                    CPU.class,
                    Broadcast.implementationForCPU()
                            .with(Fun.F64F64ToF64.triple(
                                ( a, b ) -> a * b,
                                ( a, b ) -> b, // Deriving at input 0
                                ( a, b ) -> a  // deriving input 1
                            ))
                            .with(Fun.F32F32ToF32.triple(
                                ( a, b ) -> a * b,
                                ( a, b ) -> b, // Deriving at input 0
                                ( a, b ) -> a  // deriving input 1
                            ))
                            .get()
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    Broadcast.implementationForGPU( this.getIdentifier() )
                            .with( "value = src1 * src2;\n" )
                            .and( "value += ( d == 0 ? drain : handle );\n" )
            )
        );




        //___________________________
        // TENSOR SCALAR OPERATION :

        Scalarization scalarization = new Scalarization()
                .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
                .setExecution( (caller, call) -> Result.of(CalcUtil.executeFor( caller, call, JunctionUtil::forMultiplications )).withAutoDiff(getDefaultAlgorithm()) )
                .buildFunAlgorithm();

        setAlgorithm(
            Scalarization.class,
            scalarization.setImplementationFor(
                CPU.class,
                CPUImplementation
                    .withArity(3)
                    .andImplementation(
                        call -> {
                            if ( call.getDerivativeIndex() == 0 )
                                call.setInput( 0, call.input( 2 ).shallowCopy().getUnsafe().setIsIntermediate( true ) );
                            else if ( call.getDerivativeIndex() == 1 )
                                call.setInput( 0, call.input( 1 ).shallowCopy().getUnsafe().setIsIntermediate( true ) );
                            else
                                Scalarization.implementationForCPU()
                                    .with(Fun.F64F64ToF64.triple(
                                        ( a, b ) -> a * b,
                                        ( a, b ) -> b, // Deriving at input 0
                                        ( a, b ) -> a  // deriving input 1
                                    ))
                                    .with(Fun.F32F32ToF32.triple(
                                        ( a, b ) -> a * b,
                                        ( a, b ) -> b, // Deriving at input 0
                                        ( a, b ) -> a  // deriving input 1
                                    ))
                                    .with(Fun.I32I32ToI32.triple(
                                        ( a, b ) -> a * b,
                                        ( a, b ) -> b, // Deriving at input 0
                                        ( a, b ) -> a  // deriving input 1
                                    ))
                                    .get()
                                    .run( call );
                        }
                    )
            )
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
                                call.setInput( 0, call.input( 2 ).shallowCopy().getUnsafe().setIsIntermediate( true ) );
                            else if ( call.getDerivativeIndex() == 1 )
                                call.setInput( 0, call.input( 1 ).shallowCopy().getUnsafe().setIsIntermediate( true ) );
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
                        }
                    )
                    .build()
            )
        );

    }


    @Contract(pure = true)
    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" * ");
            }
        }
        return "(" + reconstructed + ")";
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

    @Contract(pure = true)
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
