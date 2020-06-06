import neureka.Neureka

Neureka.instance {

    settings {

        debug {
            it.isKeepingDerivativeTargetPayloads = false
        }

        autoDiff {
            it.isRetainingPendingErrorForJITProp = true
            it.isApplyingGradientWhenTensorIsUsed = true
            it.isApplyingGradientWhenRequested = true
        }

        indexing {
            it.isUsingLegacyIndexing = false
            it.isUsingThoroughIndexing = true
        }

        view {
            it.isUsingLegacyView = false
        }

        ndim {
            it.isOnlyUsingDefaultNDConfiguration = false
        }

    }

    return "0.2.3"

}