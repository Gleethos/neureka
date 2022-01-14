import neureka.Neureka
import neureka.dtype.custom.F64

Neureka.configure {

    settings {

        debug {
            it.isKeepingDerivativeTargetPayloads = false
        }

        autograd {
            it.isPreventingInlineOperations = true
            it.isRetainingPendingErrorForJITProp = false
            it.isApplyingGradientWhenTensorIsUsed = false
            it.isApplyingGradientWhenRequested = false
        }

        view { 
            tensors {
                it.rowLimit          = 50
                it.isScientific      = true
                it.isMultiline       = true
                it.hasSlimNumbers    = true
                it.hasGradient       = true
                it.cellSize          = 6
                it.hasValue          = true
                it.hasRecursiveGraph = false
                it.hasDerivatives    = false
                it.hasShape          = true
                it.isCellBound       = false
                it.postfix           = ""
                it.prefix            = ""
                it.isLegacy          = false
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

    return "0.12.0"

}