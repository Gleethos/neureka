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
                                    settings.rowLimit          = 50
                                    settings.isScientific      = true
                                    settings.isMultiline       = true
                                    settings.hasSlimNumbers    = true
                                    settings.hasGradient       = true
                                    settings.cellSize          = 6
                                    settings.hasValue          = true
                                    settings.hasRecursiveGraph = false
                                    settings.hasDerivatives    = false
                                    settings.hasShape          = true
                                    settings.isCellBound       = false
                                    settings.postfix           = ""
                                    settings.prefix            = ""
                                    settings.isLegacy          = false
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