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
                                    .setRowLimit(50)
                                    .setIsScientific(true)
                                    .setIsMultiline(true)
                                    .setHasSlimNumbers(true)
                                    .setHasGradient(true)
                                    .setCellSize(6)
                                    .setHasValue(true)
                                    .setHasRecursiveGraph(false)
                                    .setHasDerivatives(false)
                                    .setHasShape(true)
                                    .setIsCellBound(false)
                                    .setPostfix("")
                                    .setPrefix("")
                                    .setIsLegacy(false)
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