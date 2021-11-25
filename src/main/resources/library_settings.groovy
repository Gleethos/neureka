import neureka.Neureka
import neureka.dtype.custom.F64
import neureka.view.Configuration
import neureka.view.TsrAsString

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

            it.asString = {
                Configuration settings ->
                                    settings
                                    .setShortage(50)
                                    .setIsCompact(true)
                                    .setIsFormatted(true)
                                    .sethaveSlimNumbers(true)
                                    .setHasGradient(true)
                                    .setPadding(6)
                                    .setHasValue(true)
                                    .setHasRecursiveGraph(false)
                                    .setHasDerivatives(false)
                                    .setHasShape(true)
                                    .setIsCellBound(false)
                                    .setPostfix("")
                                    .setPrefix("")
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