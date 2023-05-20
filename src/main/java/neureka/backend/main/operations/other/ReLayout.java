package neureka.backend.main.operations.other;

import neureka.Neureka;
import neureka.Tensor;
import neureka.backend.api.Algorithm;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Util;
import neureka.backend.main.memory.MemUtil;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.ndim.config.NDConfiguration;

import java.util.function.Supplier;

public class ReLayout extends AbstractOperation
{
    public ReLayout()
    {
        super(
            new OperationBuilder()
            .identifier(       "layout"  )
            .operator(         "layout"  )
            .arity(            1          )
            .isOperator(       false      )
            .isIndexer(        false      )
            .isDifferentiable( true       )
            .isInline(         false      )
        );
        setAlgorithm(
            Algorithm
            .withName( "layout" )
            .setIsSuitableFor( call -> SuitabilityPredicate.GOOD )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setExecution(
                ( caller, call ) ->
                {
                    Tensor<?>[] inputs = AbstractDeviceAlgorithm.flatten(caller, call).inputs();
                    Tensor<Object> input = (Tensor<Object>) inputs[0];

                    NDConfiguration.Layout originalLayout = input.getNDConf().getLayout();
                    NDConfiguration.Layout newLayout = call.getValOf( Arg.Layout.class );

                    Tensor<?> reLayout = toLayout( input.deepCopy(), newLayout );

                    return Result.of(reLayout.mut().setIsIntermediate(true))
                            .withADAction( target -> {
                                Tensor<Object> error = (Tensor<Object>) target.error().deepCopy();
                                return error.mut().toLayout(originalLayout);
                            });
                }
            )
            .buildFunAlgorithm()
        );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }


    public static Tensor<?> toLayout(Tensor<?> t, NDConfiguration.Layout target )
    {
        if ( target == t.getNDConf().getLayout() ) return t;
        if ( target == NDConfiguration.Layout.SYMMETRIC )
            throw new UnsupportedOperationException(
                    "Conversion of a non-symmetric tensor to a symmetric tensor is not possible!"
            );
        if ( target == NDConfiguration.Layout.UNSPECIFIC )
            throw new UnsupportedOperationException(
                    "Conversion of a tensor to an unspecific layout is not possible!"
            );

        if ( target == NDConfiguration.Layout.ROW_MAJOR || target == NDConfiguration.Layout.COLUMN_MAJOR ) {
            if ( t.getNDConf().getLayout() == NDConfiguration.Layout.SYMMETRIC )
                return t; // Symmetric tensors are both row and column major.
        }

        NDConfiguration old = t.getNDConf();

        if ( target == NDConfiguration.Layout.ROW_MAJOR )
            _fromCMToRM( t );
        else
            _fromRMToCM( t );

        _checkLayoutConversion( t.getNDConf(), old, target );
        return t;
    }

    /**
     *  Converts this tensor from column major to column major layout.
     */
    private static void _fromCMToRM( Tensor<?> t ) {
        if ( t.getNDConf().isVirtual() ) {
            t.mut().setIsVirtual( false ); // We actualized the tensor before conversion!
            if ( t.getNDConf().getLayout() == NDConfiguration.Layout.ROW_MAJOR )
                return;
        }
        Tensor<?> clone = t.deepCopy(); // A clone will have by default a row major layout.
        t.mut().setNDConf( clone.getNDConf() );
        _assignIfActual( t, () -> clone );
    }

    /**
     *  Converts this tensor from row major to column major layout.
     */
    private static void _fromRMToCM( Tensor<?> t ) {
        _assignIfActual( t, () -> Util.transpose(t).deepCopy().getMut().detach() );
        NDConfiguration old = t.getNDConf();
        int[] newTranslation = NDConfiguration.Layout.COLUMN_MAJOR.newStridesFor(old.shape());
        if ( old.isVirtual() ) {
            t.mut().setIsVirtual(false);
            old = t.getNDConf();
        }
        t.mut().setNDConf( _createNewNDCFrom( old, newTranslation ) );
    }

    /**
     *  This will only call the supplier and copy its result into this tensor
     *  if this tensor is not virtual (meaning this is an actual tensor).
     */
    private static void _assignIfActual(Tensor<?> t, Supplier<Tensor<?>> provider ) {
        if ( !t.isVirtual() ) {
            Tensor<?> toBeAssigned = provider.get();
            MemUtil.keep(t, toBeAssigned,
                    () -> Neureka.get().backend().getFunction().idy().execute( t, toBeAssigned )
            );
        }
    }

    private static NDConfiguration _createNewNDCFrom(
            NDConfiguration old, int[] newTranslation
    ) {
        assert !old.isVirtual();
        return NDConfiguration.of(
                    old.shape(), newTranslation, old.indicesMap(), old.spread(), old.offset()
                );
    }

    private static void _checkLayoutConversion(
            NDConfiguration newConf,
            NDConfiguration oldConf,
            NDConfiguration.Layout targetLayout
    ) {
        if ( newConf.isVirtual() )
            throw new IllegalStateException("Layout conversion produced a virtual nd-configuration!");
        if ( !newConf.getLayout().isCompatible(targetLayout) )
            throw new IllegalArgumentException(
                    "Failed to convert this tensor from its original layout '"+oldConf.getLayout()+"' " +
                            "to target layout '"+targetLayout+"'. Instead this tensor has layout '"+newConf.getLayout()+"'."
            );
    }

}
