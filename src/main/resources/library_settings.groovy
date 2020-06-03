import neureka.Neureka

Neureka.instance {

    settings {

        autoDiff {
            it.isRetainingPendingErrorForJITProp = true
            it.isApplyingGradientWhenTensorIsUsed = true
            it.isApplyingGradientWhenRequested = true
        }

        indexing {
            it.isUsingLegacyIndexing = false
            it.isUsingThoroughIndexing = true
        }

        debug {
            it.isKeepingDerivativeTargetPayloads = false
        }

        view {
            it.isUsingLegacyView = false
        }

        ndim {
            it.isOnlyUsingDefaultNDConfiguration = false
        }

    }

    return "0.2.2"

}