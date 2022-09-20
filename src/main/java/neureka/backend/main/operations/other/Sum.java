package neureka.backend.main.operations.other;

import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.DeviceAlgorithm;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.SumAlgorithm;
import neureka.backend.main.operations.linear.internal.opencl.CLSum;
import neureka.backend.main.operations.other.internal.CPUSum;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

public class Sum extends AbstractOperation
{
    public Sum()
    {
        super(
            new OperationBuilder()
                .identifier(       "sumItems"  )
                .operator(         "sumItems"  )
                .arity(            1           )
                .isOperator(       false       )
                .isIndexer(        false       )
                .isDifferentiable( true        )
                .isInline(         false       )
        );

        setAlgorithm(
            new SumAlgorithm()
        );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}
