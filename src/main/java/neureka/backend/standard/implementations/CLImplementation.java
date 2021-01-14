package neureka.backend.standard.implementations;

import lombok.Getter;
import lombok.experimental.Accessors;
import neureka.Neureka;
import neureka.backend.api.implementations.AbstractImplementationFor;
import neureka.backend.api.implementations.ImplementationFor;
import neureka.backend.api.operations.AbstractOperation;
import neureka.devices.opencl.OpenCLDevice;

import java.util.HashMap;
import java.util.Map;

/**
 * This class is the ExecutorFor &lt; OpenCLDevice &gt; implementation
 * used to properly call an OpenCLDevice instance via the
 * ExecutionOn &lt; OpenCLDevice &gt; lambda implementation
 * receiving an instance of the ExecutionCall class.
 */
@Accessors( prefix = {"_"} )
public class CLImplementation extends AbstractImplementationFor<OpenCLDevice>
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

    @Getter private String _source;
    @Getter private String _name;

    public CLImplementation(
            ImplementationFor<OpenCLDevice> lambda,
            int arity
    ) {
        super( lambda, arity );
    }

    public CLImplementation(
            ImplementationFor<OpenCLDevice> lambda,
            int arity,
            String kernelName,
            String kernelSource
    ) {
        super( lambda, arity );
        _name = kernelName;
        _source = kernelSource;
    }

    public CLImplementation(
            ImplementationFor<OpenCLDevice> lambda,
            int arity,
            String kernelSource,
            String activationSource,
            String differentiationSource,
            AbstractOperation type
    ) {
        super( lambda, arity );
        kernelSource = kernelSource.replace(
                "Neureka.instance().settings().indexing().REVERSE_INDEX_TRANSLATION",
                (Neureka.instance().settings().indexing().isUsingLegacyIndexing()) ? "true" : "false"
        );
        boolean templateFound;
        if (kernelSource.contains("__kernel")) {
            String[] parts = kernelSource.split("__kernel")[ 1 ].split("\\(")[ 0 ].split(" ");

            templateFound = parts[parts.length - 1].contains("template");
            if (!templateFound) {
                throw new IllegalStateException("Invalid source code passed to AbstractCLExecution!");
            } else {
                Map<String, String> map = _getParsedKernelsFromTemplate(
                        parts[parts.length - 1],
                        kernelSource,
                        activationSource,
                        differentiationSource,
                        type
                );
                _name = map.keySet().toArray(new String[ 0 ])[ 0 ];
                _source = map.values().toArray(new String[ 0 ])[ 0 ];
            }
        }
    }

    private interface Parser
    {
        void apply(String name, String first, String second);
    }

    private Map<String, String> _getParsedKernelsFromTemplate(
            String templateName,
            String kernelSource,
            String activationSource,
            String differentiationSource,
            AbstractOperation type
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
                type.getFunction(),
                activationSource,
                differentiationSource
        );
        return code;
    }


}