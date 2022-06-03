package neureka.backend.main.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.fun.AutoDiffMode;
import neureka.backend.api.algorithms.fun.Result;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.Operator;
import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.implementations.CLImplementation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.internal.RecursiveExecutor;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Power extends AbstractOperation
{
    public Power()
    {
        super(
                new OperationBuilder()
                        .setIdentifier(         "power"    )
                        .setOperator(         "^"        )
                        .setArity(            -1         )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( true       )
                        .setIsInline(         false      )
        );

        //_____________________
        // DEFAULT OPERATION :

        RecursiveExecutor rja = (call, traverse)->
        {
            Device<Number> device = call.getDeviceFor(Number.class);
            int d = call.getValOf( Arg.DerivIdx.class );
            Operation type = call.getOperation();

            Tsr<?> result = null;
            if ( call.arity() > 3 )
            {
                if ( d < 0 ) {
                    Tsr<?>[] reduction = new Tsr[]{call.input( 0 ), call.input( 1 ), call.input( 2 )};
                    call.setInput( 0, traverse.execute( call.withInputs( reduction ) ) );
                    reduction = Utility.offsetted( call.inputs(), 1 );
                    result = traverse.execute( call.withInputs( reduction ) );
                    call.setInput( 0, result );
                } else {

                    Tsr<?>[] reduction = Utility.subset(call.inputs(), 1,  2, call.arity()-2);
                    reduction[ 0 ] = call.input( 1 ).clone().getUnsafe().setIsIntermediate( true );

                    if ( d==0 ) {
                        Tsr<?> exp = traverse.execute(
                                                ExecutionCall.of( reduction )
                                                            .andArgs(Arg.DerivIdx.of( -1 ))
                                                            .running(Neureka.get().backend().getOperation("*"))
                                                            .on(device)
                                            );

                        reduction = new Tsr[]{call.input( 0 ), call.input( 1 ), exp};
                        call.setInput( 0, traverse.execute(
                                                    ExecutionCall.of( reduction )
                                                                    .andArgs( Arg.DerivIdx.of(0) )
                                                                    .running(type)
                                                                    .on( device )
                                                ));
                        exp.getUnsafe().delete();
                    } else {
                        Tsr<?> inner = traverse.execute(
                                                ExecutionCall.of( reduction )
                                                                .andArgs(Arg.DerivIdx.of(d-1))
                                                                .running(Neureka.get().backend().getOperation("*"))
                                                                .on(device)
                                        );

                        reduction = new Tsr[]{ call.input( 1 ).clone().getUnsafe().setIsIntermediate( true ), inner, call.input( d ) };
                        Tsr<?> exp = traverse.execute(
                                                ExecutionCall.of( reduction )
                                                                .andArgs( Arg.DerivIdx.of(-1) )
                                                                .running( Neureka.get().backend().getOperation("*") )
                                                                .on( device )
                                      );

                        reduction = new Tsr[]{call.input( 0 ), call.input( 1 ), exp};
                        result = traverse.execute(
                                                ExecutionCall.of( reduction )
                                                            .andArgs(Arg.DerivIdx.of(1))
                                                            .running(type)
                                                            .on(device)
                                            );

                        call.setInput( 0, result );

                        inner.getUnsafe().delete();
                        exp.getUnsafe().delete();
                    }
                }
            }
            return result;
        };

        Operator operator = new Operator( rja )
                                .setSupplyADAgentFor( getDefaultAlgorithm() )
                                .buildFunAlgorithm();

        setAlgorithm(Operator.class,
            operator.setImplementationFor(
                CPU.class,
                Operator.implementationForCPU()
                    .with(Fun.F64F64ToF64.triple(
                        ( a, b ) -> Math.pow( a, b ),
                        ( a, b ) -> b * Math.pow( a, b - 1 ), // Deriving at input 0
                        ( a, b ) -> Math.pow( a, b ) * Math.log( a ) // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> (float) Math.pow( a, b ),
                        ( a, b ) -> (float) (b * Math.pow( a, b - 1 )), // Deriving at input 0
                        ( a, b ) -> (float) (Math.pow( a, b ) * Math.log( a )) // deriving input 1
                    ))
                    .with(Fun.I32I32ToI32.triple(
                        ( a, b ) -> (int) Math.round(Math.pow( a, b )),
                        ( a, b ) -> (int) Math.round(b * Math.pow( a, b - 1 )), // Deriving at input 0
                        ( a, b ) -> (int) Math.round(Math.pow( a, b ) * Math.log( a )) // deriving input 1
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                Operator.implementationForGPU( this.getIdentifier() )
                        .with( "output = pow(input1, input2);" )
                        .and(
                            "if ( d == 0 ) {                                    \n" +
                                    "    output = input2 * pow(input1, input2-1.0f);  \n" +
                                    "} else {                                         \n" +
                                    "    output = pow(input1, input2) * log(input1);  \n" +
                                    "}"
                        )
            )
        );

        //________________
        // BROADCASTING :

        Broadcast broadcast = new Broadcast(rja)
                .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
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
                .buildFunAlgorithm();

        setAlgorithm(
            Broadcast.class,
            broadcast.setImplementationFor(
                CPU.class,
                Broadcast.implementationForCPU()
                    .with(Fun.F64F64ToF64.triple(
                        ( a, b ) -> Math.pow(a, b),
                        // In the context of broadcasting the traditional scalar derivative would be 1, broadcasting has different rules...
                        ( a, b ) -> a * Math.pow( a, b - 1  ), // Deriving at input 0
                        ( a, b ) -> Math.pow( a, b ) * Math.log(a) // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> (float) Math.pow(a, b),
                        // In the context of broadcasting the traditional scalar derivative would be 1, broadcasting has different rules...
                        ( a, b ) -> (float) (a * Math.pow( a, b - 1  )), // Deriving at input 0
                        ( a, b ) -> (float) (Math.pow( a, b ) * Math.log(a)) // deriving input 1
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                Broadcast.implementationForGPU( this.getIdentifier() )
                        .with( "value += pow(src1, src2);" )
                        .and(
                            "if ( d == 0 ) {\n" +
                            "    value = (handle * pow(target, handle-(float)1 )) * drain;\n" +
                            "} else {\n" +
                            "    value += (pow(target, handle) * log(handle)) * drain;\n" +
                            "}"
                        )
            )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        Scalarization scalarization =
            new Scalarization()
                .setIsSuitableFor( call -> SuitabilityPredicate.BAD )
                .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
                .setExecution( (caller, call) -> Result.of(CalcUtil.executeFor( caller, call, rja )).withAutoDiff(getDefaultAlgorithm()) )
                .buildFunAlgorithm();

        setAlgorithm(
            Scalarization.class,
            scalarization.setImplementationFor(
                CPU.class,
                Scalarization.implementationForCPU()
                    .with(Fun.F64F64ToF64.triple(
                        ( a, b ) -> Math.pow( a, b ),
                        ( a, b ) -> b * Math.pow( a, b - 1 ), // Deriving at input 0
                        ( a, b ) -> Math.pow( a, b ) * Math.log( a ) // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> (float) Math.pow( a, b ),
                        ( a, b ) -> (float) (b * Math.pow( a, b - 1 )), // Deriving at input 0
                        ( a, b ) -> (float) (Math.pow( a, b ) * Math.log( a )) // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> (int) Math.round(Math.pow( a, b )),
                        ( a, b ) -> (int) Math.round(b * Math.pow( a, b - 1 )), // Deriving at input 0
                        ( a, b ) -> (int) Math.round(Math.pow( a, b ) * Math.log( a )) // deriving input 1
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                CLImplementation
                    .compiler()
                    .arity( 3 )
                    .kernelSource( scalarization. getKernelSource() )
                    .activationSource( "output = pow( input1, value );" )
                    .differentiationSource(
                        "if ( d == 0 ) {                                      \n" +
                        "    output = value * pow( input1, value - (float) 1 );   \n" +
                        "} else {                                             \n" +
                        "    output = pow( input1, value ) * log( value );        \n" +
                        "}"
                    )
                    .kernelPostfix( this.getIdentifier() )
                    .execution(
                        call -> {
                            int offset = (call.input( Number.class, 2 ).isVirtual() || call.input( Number.class, 2 ).size() == 1)?1:0;
                            int gwz = call.input( Number.class, 0 ).size();
                            call.getDevice()
                                .getKernel( call )
                                .passAllOf(call.input( Number.class, 0 ))
                                .passAllOf(call.input( Number.class, 0 ))
                                .pass( call.input( Number.class, 1 + offset ).at( 0 ).get().floatValue() )
                                .pass( call.input( Number.class, 0 ).rank() )
                                .pass( call.getValOf( Arg.DerivIdx.class ) )
                                .call( gwz );
                        }
                    )
                    .build()
            )
        );


    }

    // d/dx(f(x)^g(x))=
    // f(x)^g(x) * d/dx(g(x)) * ln(f(x))
    // + f(x)^(g(x)-1) * g(x) * d/dx(f(x))
    @Contract(pure = true)

    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) reconstructed.append(" ^ ");
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        Function a = children[0];
        Function b = Function.of(
                IntStream.range( 1, children.length )
                .mapToObj(i -> children[ i ].toString() )
                .collect(Collectors.joining(" * "))
        );
        boolean aDerivable = a.dependsOn(derivationIndex);
        boolean bDerivable = b.dependsOn(derivationIndex);
        String aAsStr = a.toString();
        String bAsStr = b.toString();
        String first = "";
        if (aDerivable) {
            String aAsDerivative = a.getDerivative(derivationIndex).toString();
            if ( !aAsDerivative.equals("0.0") ) {
                first = ("( "+ bAsStr +" * "+ aAsStr + " ^ (" + bAsStr + " - 1) )");
                if (!aAsDerivative.equals("1.0")) first = aAsDerivative + " * " + first;
            }
        }
        String bAsDerivative = "";
        if (bDerivable) bAsDerivative = b.getDerivative(derivationIndex).toString();
        if ( !bAsDerivative.isEmpty() && !bAsDerivative.equals("1.0") ) bAsDerivative += " * ";
        else bAsDerivative = "";
        String second = "";
        if ( bDerivable ) second = "(ln("+aAsStr+") * "+aAsStr+" ^ "+bAsStr+")";
        String result;
        if ( !first.trim().isEmpty() && !second.trim().isEmpty() ) result = bAsDerivative+"("+first+" + "+second+")";
        else if (!first.trim().isEmpty()) result = bAsDerivative + "("+first+")";
        else if (!second.trim().isEmpty()) result = bAsDerivative + "(" +second + ")";
        else result = bAsDerivative;
        return result;
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result = Math.pow(result, current);
            }
            return result;
        } else {
            double b = 1;
            double bd = 0;
            double a;
            for ( int i = 1; i < src.length; i++ ) {
                double dd = 1;
                a = src[ i ].call( inputs, j );
                for ( int di = 1; di < src.length; di++ ) {
                    if ( di != i ) dd *= a;
                    else dd *= src[ di ].derive( inputs, d, j );
                }
                bd += dd;
                b *= a;
            }
            double out = 0;
            a = src[ 0 ].call( inputs, j );
            out += src[ 0 ].derive( inputs, d, j ) * b * Math.pow(a, b - 1);
            out += (a >= 0) ? bd *  Math.pow(a, b) * Math.log(a) : 0;
            return out;
        }
    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result = Math.pow(result, current);
            }
            return result;
        } else {
            double b = 1;
            double bd = 0;
            double a;
            for ( int i = 1; i < src.length; i++ ) {
                double dd = 1;
                a = src[ i ].call( inputs );
                for ( int di = 1; di < src.length; di++ ) {
                    if ( di != i ) dd *= a;
                    else dd *= src[ di ].derive( inputs, d );
                }
                bd += dd;
                b *= a;
            }
            double out = 0;
            a = src[ 0 ].call( inputs );
            out += src[ 0 ].derive( inputs, d ) * b * Math.pow(a, b - 1);
            out += (a >= 0) ? bd *  Math.pow(a, b) * Math.log(a) : 0;
            return out;
        }
    }







}
