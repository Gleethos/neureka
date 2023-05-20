package neureka.backend.main.implementations.broadcast;

import neureka.Neureka;
import neureka.Tensor;
import neureka.backend.main.implementations.ParsedCLImplementation;
import neureka.math.args.Arg;
import neureka.devices.opencl.KernelCode;
import neureka.dtype.DataType;

import java.util.Arrays;

public class CLScalarBroadcast extends ParsedCLImplementation
{
    protected final static String TYPE = "#DATA_TYPE#";

    public CLScalarBroadcast(
        String postfix, String activation, String derivation
    ) {
        super(
            call->{
                Tensor<Number> t = call.input( Number.class, 0 );
                int gwz = t.size();
                call.getDevice()
                        .getKernel(call)
                        .passAllOf( t )
                        .passAllOf( t )
                        .pass( call.input( Number.class, 1 ).at(0).get().floatValue() )
                        .pass( t.rank() )
                        .pass( call.getValOf( Arg.DerivIdx.class ) )
                        .call( gwz );

                return call.input(0);
            },
            2,
            Neureka.get().utility().readResource("kernels/scalarization_template.cl"),
            activation,
            derivation,
            postfix,
            kernelCode -> {
                String[] types = new String[]{
                        "float", "double", "int", "long", "short", "char"
                };
                return
                    Arrays.stream(types).map( type -> {
                        String newName = kernelCode.getName() + ("_" + type);
                        String newCode = kernelCode.getCode()
                                                    .replace(TYPE, type)
                                                    .replace(kernelCode.getName(), newName);
                        DataType<?> dt;
                        switch (type) {
                            case "float":  dt = DataType.of(Float.class);   break;
                            case "double": dt = DataType.of(Double.class);  break;
                            case "int":    dt = DataType.of(Integer.class); break;
                            case "long":   dt = DataType.of(Long.class);    break;
                            case "short":  dt = DataType.of(Short.class);   break;
                            case "char":   dt = DataType.of(Byte.class);    break;
                            default:       dt = DataType.of(Float.class);   break;
                        }
                        return new KernelCode(newName, newCode, dt);
                    })
                    .toArray(KernelCode[]::new);
            }
        );
    }
}
