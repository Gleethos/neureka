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
            tensors {
                TsrStringSettings settings ->
                                    settings
                                    .withRowLimit(50)
                                    .scientific(true)
                                    .multiline(true)
                                    .withSlimNumbers(true)
                                    .withGradient(true)
                                    .withPadding(6)
                                    .withValue(true)
                                    .withRecursiveGraph(false)
                                    .withDerivatives(false)
                                    .withShape(true)
                                    .cellBound(false)
                                    .withPostfix("")
                                    .withPrefix("")
                                    .legacy(false)
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