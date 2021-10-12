import neureka.Neureka
import neureka.dtype.custom.F64
import neureka.utility.TsrAsString

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
            
            it.asString = [
                    (TsrAsString.Should.BE_SHORTENED_BY)      : 50,
                    (TsrAsString.Should.BE_COMPACT)           : true,
                    (TsrAsString.Should.BE_FORMATTED)         : true,
                    (TsrAsString.Should.HAVE_GRADIENT)        : true,
                    (TsrAsString.Should.HAVE_PADDING_OF)      : 6,
                    (TsrAsString.Should.HAVE_VALUE)           : true,
                    (TsrAsString.Should.HAVE_RECURSIVE_GRAPH) : false,
                    (TsrAsString.Should.HAVE_DERIVATIVES)     : false,
                    (TsrAsString.Should.HAVE_SHAPE)           : true,
                    (TsrAsString.Should.BE_CELL_BOUND)        : false
            ]
        }

        ndim {
            it.isOnlyUsingDefaultNDConfiguration = false
        }

        dtype {
            it.defaultDataTypeClass = F64.class
            it.isAutoConvertingExternalDataToJVMTypes = true
        }

    }

    return "0.8.0"

}