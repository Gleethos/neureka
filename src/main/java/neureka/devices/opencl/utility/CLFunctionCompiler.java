package neureka.devices.opencl.utility;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.DefaultADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.GenericAlgorithm;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.calculus.implementations.FunctionInput;
import neureka.calculus.implementations.FunctionVariable;
import neureka.devices.Device;
import neureka.devices.opencl.KernelCaller;
import neureka.devices.opencl.OpenCLDevice;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class CLFunctionCompiler {

    private final OpenCLDevice _device;
    private final Function _functionToBeOptimized;
    private final String _functionName;
    private final int[] _argPointer;

    public CLFunctionCompiler( OpenCLDevice device, Function toBeOptimized, String functionName ) {
        _device = device;
        _functionToBeOptimized = toBeOptimized;
        _functionName = functionName;
        _argPointer = toBeOptimized.getAllNodes()
                                    .stream()
                                    .filter( fun -> fun instanceof FunctionInput )
                                    .mapToInt( fun -> ( (FunctionInput) fun ).index() )
                                    .distinct()
                                    .toArray();
    }


    public Operation optimize() {
        int numberOfArgs = _functionToBeOptimized.numberOfArgs();
        if ( _functionToBeOptimized.getNodes().stream().anyMatch(fun -> fun instanceof FunctionVariable ) )
            numberOfArgs = -1; // The function is an indexer which means that it can have any number of arguments...
        return new OperationBuilder()
                .setFunction(_functionName)
                .setOperator(_functionName)
                .setArity( numberOfArgs )
                .setIsIndexer( numberOfArgs < 0 )
                .setIsOperator( false )
                .setIsDifferentiable( true )
                .setIsInline( false )
                .setStringifier(
                        children -> {
                            String expression = String.join(", ", children);
                            if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                                return _functionName + expression;
                            }
                            return _functionName + "(" + expression + ")";
                        }
                )
                .build()
                .setAlgorithm(
                        new GenericAlgorithm( "generic_algorithm_for_"+ _functionName)
                                .setIsSuitableFor( call -> 1.0f )
                                .setCanPerformBackwardADFor( call -> true )
                                .setCanPerformForwardADFor( call -> false )
                                .setSupplyADAgentFor(
                                        (Function f, ExecutionCall<? extends Device<?>> call, boolean forward) -> {
                                            // TODO: calculate derivative and supply agent!
                                            return new DefaultADAgent(null)
                                                    .setForward((t, derivative) -> new FunctionBuilder(Neureka.get().context()).build(f.toString(), false).derive(new Tsr[]{derivative}, 0))
                                                    .setBackward((t, error) -> new FunctionBuilder(Neureka.get().context()).build(f.toString(), false).derive(new Tsr[]{error}, 0));
                                        }
                                )
                                .setHandleInsteadOfDevice( (caller, call) -> null )
                                .setHandleRecursivelyAccordingToArity( (call, goDeeperWith) -> null )
                                .setInstantiateNewTensorsForExecutionIn(
                                        call -> {
                                            Tsr<?>[] args = call.getTensors();
                                            if ( args[0] == null ) // Creating a new tensor:
                                            {
                                                Tsr<Double> output = Tsr.of(args[1].getNDConf().shape(), 0.0);
                                                output.setIsVirtual( false );
                                                call.getDeviceFor(Double.class).store(output);
                                                args[0] = output;
                                            }
                                            return call;
                                        }
                                )
                                .setImplementationFor(
                                        OpenCLDevice.class,
                                        this::_adHocKernelFor
                                )
                );
    }


    private void _adHocKernelFor( ExecutionCall<?> call ) {

        List<Tsr<Number>> args = Arrays.stream( _argPointer )
                                    .mapToObj( p -> (Tsr<Number>) call.getTensors()[p+1] )
                                    .collect(Collectors.toList());

        args.add(0, call.getTsrOfType(Number.class, 0));

        List<String> types = args.stream()
                                    .map( CLFunctionCompiler::_clTypeOf )
                                    .collect(Collectors.toList());

        String kernelSignature =
                                _functionName + ( call.getDerivativeIndex() >= 0 ? "_derivative" : "" ) +
                                "_" +
                                IntStream
                                    .range(0, args.size())
                                    .mapToObj(
                                            i -> types.get(i) + "$" +
                                                    args.get(i)
                                                        .shape()
                                                        .stream()
                                                        .map( String::valueOf )
                                                        .collect(Collectors.joining("x"))
                                    )
                                    .collect(Collectors.joining("_"));

        if ( this._device.hasAdHocKernel( kernelSignature ) ) {
            KernelCaller caller = _device.getAdHocKernel( kernelSignature );
            args.forEach( caller::pass );
            caller.call( args.get(0).size() );
            return;
        }
        // So no kernel with this signature was found...
        // Therefore we compile a new kernel specific to the provided call contents (shapes and types)!

        int rank = args.get(0).rank();

        List<List<String>> configs = args.stream()
                                            .map( arg -> arg.getNDConf().asInlineArray() )
                                            .map(
                                                    array ->
                                                            Arrays.stream(array)
                                                                    .mapToObj( String::valueOf )
                                                                    .collect(Collectors.toList())
                                            )
                                            .collect(Collectors.toList());

        String argString = IntStream.range( 0, args.size() )
                                    .mapToObj( i -> "__global "+types.get(i)+"* arg" + i )
                                    .collect(Collectors.joining(", "));

        Function toBeCompiled = call.getDerivativeIndex() < 0
                                    ? _functionToBeOptimized
                                    : _functionToBeOptimized.getDerivative( call.getDerivativeIndex() );

        String compilableFun = IntStream.range( 0, _argPointer.length )
                                        .mapToObj( String::valueOf )
                                        .reduce(
                                            toBeCompiled.toString(),
                                             (source, index) ->
                                                     source.replace(
                                                             "I["+_argPointer[Integer.parseInt(index)]+"]",
                                                             "v" + (index + 1)
                                                     )
                                        );

        String kernelCode =
                "\n\n" +
                "    __kernel void " + kernelSignature + "(\n" +
                "        " + argString + "\n" +
                "    ) {                                                                                     \n" +
                "        " + IntStream
                                .range(0, configs.size())
                                .mapToObj(
                                    i -> "int* cfg"+i+" = {" + String.join( ",", configs.get(i) ) + "};"
                                )
                                .collect(Collectors.joining("\n        ")) +
                "                                                                                          \n" +
                "        " + IntStream
                                .range(1, args.size()) // We start at 1 because 0 is the output!
                                .mapToObj(
                                        i -> types.get(i) + " v" + i + " = arg" + i + "[_i_of_i(i, cfg"+i+", "+rank+")];"
                                )
                                .collect(Collectors.joining("\n        ")) +
                "                                                                                          \n" +
                "        unsigned int i = get_global_id( 0 );                                              \n" +
                "        arg0[_i_of_i(i, cfg0, "+rank+")] = " + compilableFun + ";                         \n" +
                "    }                                                                                     \n\n" +
                "    " + _readIndexMapper();

        _device.compileAdHocKernel( kernelSignature, kernelCode );
        KernelCaller caller = _device.getAdHocKernel( kernelSignature );
        args.forEach( caller::pass );
        caller.call( args.get(0).size() );
    }

    private static String _clTypeOf( Tsr<?> tensor ) {
        DataType<?> dtype = tensor.getDataType();
        java.util.function.Function<Class<?>, String> formatter = type -> type.getSimpleName()
                                                                                 .toLowerCase()
                                                                                 .replace("integer", "int");
        if (dtype.typeClassImplements(NumericType.class) ) {
            NumericType<?,?,?,?> instance = (NumericType) dtype.getTypeClassInstance();
            if ( instance.holderType() == instance.targetType() )
                return formatter.apply(instance.holderType()); // Float, Double, Long, Short...
            else // Unsigned types:
                return "u" + formatter.apply(instance.holderType());
        }
        return formatter.apply(dtype.getTypeClass());
    }

    private static String _readIndexMapper() {
        return "int _i_of_i" +
               Neureka.get()
                        .utility()
                        .readResource("kernels/utility.cl")
                        .split("int _i_of_i")[1]
                        .split("// _i_of_i end!")[0];
    }

}
