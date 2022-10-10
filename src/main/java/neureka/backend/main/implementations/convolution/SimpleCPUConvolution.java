package neureka.backend.main.implementations.convolution;

import neureka.Tsr;
import neureka.devices.host.CPU;
import neureka.ndim.config.NDConfiguration;

/**
 *  Performs fast image convolution on nd-array of rank 2 (matrices) or rank 3 (tensors with batch size)
 *  where one tensor is the kernel and the other one is the image.
 */
class SimpleCPUConvolution
{
    Conv2DImpl _impl;

    SimpleCPUConvolution( Tsr<?> in1, Tsr<?> in2, Tsr<?> out ) {
        Conv2DImpl impl = null;
        try {
            impl = _tryCreatingImplFor( in1, in2, out );
        }
        catch ( Exception ignored ) {}
        try {
            impl = _tryCreatingImplFor( in2, in1, out );
        }
        catch ( Exception ignored ) {}
        _impl = impl;
    }

    public void run() {
        if ( _impl == null ) throw new IllegalStateException("Not runnable!");
        _impl.run();
    }

    public boolean isSuitable() { return _impl != null; }

    private static Conv2DImpl _tryCreatingImplFor(
            final Tsr<?> image,
            final Tsr<?> kernel,
            final Tsr<?> result
    ) {
        validate(image);
        validate(kernel);
        validate(result);
        int batchSize = (image.rank() == 3 ? image.shape(0) : 1);
        int shapeOffset = (image.rank() == 3 ? 1 : 0);
        int width = image.shape(shapeOffset + 1);
        int height = image.shape(shapeOffset + 0);
        int kernelWidth = kernel.shape(shapeOffset + 1);
        int kernelHeight = kernel.shape(shapeOffset + 0);
        int kernelBatchSize = (kernel.rank() == 3 ? kernel.shape(0) : 1);
        int resultWidth = width - kernelWidth + 1;
        int resultHeight = height - kernelHeight + 1;

        if ( kernelBatchSize > 1 )
            throw new IllegalArgumentException("Kernel batch size must be 1!");

        if ( batchSize * resultHeight * resultWidth != result.size() )
            throw new IllegalArgumentException("The result array must have the same length as the batch size times the result height times the result width!");

        Class<?> c1 = image.itemType();
        Class<?> c2 = kernel.itemType();
        Class<?> c3 = result.itemType();

        if ( c1 != c2 || c2 != c3 )
            throw new IllegalArgumentException("All inputs must be of the same type!");

        if ( c1 == Float.class )
            return new ImplF32(
                    image.getMut().getDataAs(float[].class),
                    kernel.getMut().getDataAs(float[].class),
                    result.getMut().getDataForWriting(float[].class),
                    width,
                    height,
                    kernelWidth,
                    kernelHeight,
                    resultWidth,
                    resultHeight,
                    batchSize
                );
        else if ( c1 == Double.class )
            return new ImplF64(
                    image.getMut().getDataAs(double[].class),
                    kernel.getMut().getDataAs(double[].class),
                    result.getMut().getDataForWriting(double[].class),
                    width,
                    height,
                    kernelWidth,
                    kernelHeight,
                    resultWidth,
                    resultHeight,
                    batchSize
                );
        else
            throw new IllegalArgumentException("Unsupported data type!");
    }

    interface Conv2DImpl {
        void run();
    }

    private static class ImplF32 implements Conv2DImpl {

        private final float[] _image;
        private final float[] _kernel;
        private final float[] _result;
        private final int _width, _height, _kernelWidth, _kernelHeight, _resultWidth, _resultHeight, _batchSize;

        private ImplF32(
                float[] image,
                float[] kernel,
                float[] result,
                int width,
                int height,
                int kernelWidth,
                int kernelHeight,
                int resultWidth,
                int resultHeight,
                int batchSize
        ) {
            _image = image;
            _kernel = kernel;
            _width = width;
            _height = height;
            _kernelWidth = kernelWidth;
            _kernelHeight = kernelHeight;
            _resultWidth = resultWidth;
            _resultHeight = resultHeight;
            _batchSize = batchSize;
            if ( _batchSize * _resultHeight * _resultWidth != result.length )
                throw new IllegalArgumentException("The result array must have the same length as the batch size times the result height times the result width!");

            _result = result;
        }

        @Override
        public void run() {
            int work = _resultHeight * _resultWidth;
            if ( work < 1000 )
                for ( int bi = 0; bi < _batchSize; bi++ ) run(bi);
            else
                CPU.get().getExecutor().threaded(_batchSize, this::run);
        }

        private void run(int batchIndex) {
            int imageOffset = batchIndex * _width * _height;
            int resultOffset = batchIndex * _resultWidth * _resultHeight;
            for ( int y = 0; y < _resultHeight; y++ ) {
                for ( int x = 0; x < _resultWidth; x++ ) {
                    float sum = 0;
                    for ( int ky = 0; ky < _kernelHeight; ky++ )
                        for ( int kx = 0; kx < _kernelWidth; kx++ )
                            sum +=
                                _image[imageOffset + (y + ky) * _width + (x + kx)]
                                        *
                                _kernel[ky * _kernelWidth + kx];

                    _result[resultOffset + y * _resultWidth + x] = sum;
                }
            }
        }
    }

    private static class ImplF64 implements Conv2DImpl {

        private final double[] _image;
        private final double[] _kernel;
        private final double[] _result;
        private final int _width, _height, _kernelWidth, _kernelHeight, _resultWidth, _resultHeight, _batchSize;

        private ImplF64(
                double[] image,
                double[] kernel,
                double[] result,
                int width,
                int height,
                int kernelWidth,
                int kernelHeight,
                int resultWidth,
                int resultHeight,
                int batchSize
        ) {
            _image = image;
            _kernel = kernel;
            _width = width;
            _height = height;
            _kernelWidth = kernelWidth;
            _kernelHeight = kernelHeight;
            _resultWidth = resultWidth;
            _resultHeight = resultHeight;
            _batchSize = batchSize;
            if ( _batchSize * _resultHeight * _resultWidth != result.length )
                throw new IllegalArgumentException("The result array must have the same length as the batch size times the result height times the result width!");

            _result = result;
        }

        @Override
        public void run() {
            int work = _resultHeight * _resultWidth;
            if ( work < 1000 )
                for ( int bi = 0; bi < _batchSize; bi++ ) run(bi);
            else
                CPU.get().getExecutor().threaded(_batchSize, this::run);
        }

        private void run(int batchIndex) {
            int imageOffset = batchIndex * _width * _height;
            int resultOffset = batchIndex * _resultWidth * _resultHeight;
            for ( int y = 0; y < _resultHeight; y++ ) {
                for ( int x = 0; x < _resultWidth; x++ ) {
                    double sum = 0;
                    for ( int ky = 0; ky < _kernelHeight; ky++ )
                        for ( int kx = 0; kx < _kernelWidth; kx++ )
                            sum +=
                                _image[imageOffset + (y + ky) * _width + (x + kx)]
                                        *
                                _kernel[ky * _kernelWidth + kx];

                    _result[resultOffset + y * _resultWidth + x] = sum;
                }
            }
        }
    }


    private static void validate(Tsr<?> t) {
        if ( t.getRank() != 2 && t.getRank() != 3 )
            throw new IllegalArgumentException("The rank of the tensor must be 2 or 3!");

        NDConfiguration.Layout layout = t.getNDConf().getLayout();

        if ( layout != NDConfiguration.Layout.ROW_MAJOR && layout != NDConfiguration.Layout.SYMMETRIC )
            throw new IllegalArgumentException("The layout of the tensor must be row major or symmetric!");
    }

}
