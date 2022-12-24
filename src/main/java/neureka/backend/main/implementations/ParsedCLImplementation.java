package neureka.backend.main.implementations;

import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.devices.opencl.KernelCode;
import neureka.devices.opencl.OpenCLDevice;
import neureka.dtype.DataType;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public class ParsedCLImplementation extends CLImplementation
{

    private final java.util.function.Function<String, String> _aliasSwapper =
            s ->
                    "//-=<PARSED>=-//\n" +
                            s.replace("src1", "src1[_i_of_idx_on_tln(prv_src1_cfg, rank)]")
                                    .replace("src2", "src2[_i_of_idx_on_tln(prv_src2_cfg, rank)]")
                                    .replace("input1", "src1[_i_of_i(i, prv_src1_cfg, rank)]")
                                    .replace("input2", "src2[_i_of_i(i, prv_src2_cfg, rank)]")
                                    .replace("input", "src1[_i_of_i(i, prv_src1_cfg, rank)]")
                                    .replace("output", "drn[_i_of_i(i, prv_drn_cfg, rank)]")
                                    .replace("handle", "src1[_i_of_idx_on_tln(prv_src1_cfg, rank)]")
                                    .replace("drain", "src2[_i_of_idx_on_tln(prv_src2_cfg, rank)]")
                                    .replace("origin", "drn[di]")
                                    .replace("target", "frn[_i_of_idx_on_tln(prv_frn_cfg, rank)]") +
                            "\n//-=<PARSED>=-//";

    private final java.util.function.Function<String, String> asAdvanced =
            s ->
                    s.replace("target", "frn[_i_of_idx_on_tln(prv_frn2_cfg, rank)]")
                            .replace("input3","frn[_i_of_idx_on_tln(prv_frn2_cfg, rank)]")
                            .replace("//-=<ARGUMENT>=-//", "")
                            .replace("//-=<CONFIGURATION>=-//", "");

    private final KernelCode[] _kernels;

    public ParsedCLImplementation(
        ImplementationFor<OpenCLDevice> lambda,
        int arity,
        String kernelSource,
        String activationSource,
        String differentiationSource,
        String postfix,
        Function<KernelCode, KernelCode[]> dataTypeAdapter
    ) {
        super( lambda, arity );
        String parsedCode = null;
        String parsedName = null;
        if ( activationSource == null && differentiationSource == null ) {
            parsedCode = kernelSource;
            parsedName = postfix;
        } else if (kernelSource.contains("__kernel")) {
            boolean templateFound;
            String[] parts = kernelSource.split("__kernel")[ 1 ].split("\\(")[ 0 ].split(" ");

            templateFound = parts[parts.length - 1].contains("template");
            if (!templateFound)
                throw new IllegalStateException("Invalid source code passed to AbstractCLExecution!");
            else {
                Map<String, String> map = _getParsedKernelsFromTemplate(
                        parts[parts.length - 1],
                        kernelSource,
                        activationSource,
                        differentiationSource,
                        postfix
                );
                parsedName = map.keySet().toArray(new String[ 0 ])[ 0 ];
                parsedCode = map.values().toArray(new String[ 0 ])[ 0 ];
            }
        }
        _kernels = dataTypeAdapter.apply( new KernelCode( parsedName, parsedCode ) );
    }

    private Map<String, String> _getParsedKernelsFromTemplate(
            String templateName,
            String kernelSource,
            String activationSource,
            String differentiationSource,
            String postfix
    ) {
        Map<String, String> code = new HashMap<>();
        String preName = templateName.replace("template", "");
        String source = kernelSource.replace("template", "");
        String[] parts = source.split("//-=<OPERATION>=-//");

        Parser parser = ( n, f, s ) -> {
            String convcode =
                    parts[ 0 ].replace(preName, preName + n) +
                            _aliasSwapper.apply(f) +
                            parts[ 2 ] +
                            _aliasSwapper.apply(s) +
                            parts[4];
            boolean isAdvanced = s.contains("target")&&s.contains("drain")&&s.contains("handle")
                    || s.contains("input1")&&s.contains("input2")&&s.contains("input3");
            convcode = (isAdvanced) ? asAdvanced.apply(convcode) : convcode;
            code.put(preName + n, convcode);
        };
        //Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
        //Tsr t0_drain,  Tsr t1_src1,   Tsr t2_src2
        //drn[di], src1[_i_of_idx_on_tln(prv_src1_cfg, rank)], src2[_i_of_idx_on_tln(prv_src2_cfg, rank)]
        //default:  src1 o src2 -> drain
        //inverse:  src1/fdrn <-src2 <- drain
        //===========================================================================
        parser.apply(
                postfix,
                activationSource,
                differentiationSource
            );
        return code;
    }

    @Override
    public KernelCode getKernelFor( ExecutionCall<OpenCLDevice> call ) {
        DataType<?> callType = call.input(0 ).getDataType();
        return Arrays.stream(_kernels)
                        .filter( k -> k.getDataType().equals( callType ) )
                        .findFirst()
                        .orElse(_kernels[0]);
    }

    @Override
    public KernelCode[] getKernelCode() {
        return _kernels;
    }

    private interface Parser
    {
        void apply( String name, String first, String second );
    }

}
