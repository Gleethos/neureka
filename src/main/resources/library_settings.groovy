import neureka.Neureka
import neureka.dtype.custom.F64
import neureka.view.TsrStringSettings

Neureka.configure {

    settings {

        debug {
            it.isKeepingDerivativeTargetPayloads = false
        }

        autograd {
            it.isPreventingInlineOperations = true
            it.isRetainingPendingErrorForJITProp = true
            it.isApplyingGradientWhenTensorIsUsed = true
            it.isApplyingGradientWhenRequested = true
        }

        indexing { 
            it.isUsingArrayBasedIndexing = true
        }

        view {
            it.isUsingLegacyView = false

            tensor {
                TsrStringSettings settings ->
                                    settings
                                    .rowLimit(50)
                                    .isCompact(true)
                                    .isFormatted(true)
                                    .hasSlimNumbers(true)
                                    .hasGradient(true)
                                    .padding(6)
                                    .hasValue(true)
                                    .hasRecursiveGraph(false)
                                    .hasDerivatives(false)
                                    .hasShape(true)
                                    .isCellBound(false)
                                    .postfix("")
                                    .prefix("")
            }
        }

        ndim {
            it.isOnlyUsingDefaultNDConfiguration = false
        }

        dtype {
            it.defaultDataTypeClass = F64.class
            it.isAutoConvertingExternalDataToJVMTypes = true
        }

    }

    return "0.9.0"

}