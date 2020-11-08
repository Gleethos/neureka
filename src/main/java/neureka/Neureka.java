/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   _   _                     _
  | \ | |                   | |
  |  \| | ___ _   _ _ __ ___| | ____ _
  | . ` |/ _ \ | | | '__/ _ \ |/ / _` |
  | |\  |  __/ |_| | | |  __/   < (_| |
  |_| \_|\___|\__,_|_|  \___|_|\_\__,_|

    This is a central singleton class used to configure the Neureka library.

*/

package neureka;

import neureka.dtype.custom.F64;
import neureka.utility.SettingsLoader;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class Neureka
{
    private static final ThreadLocal<Neureka> _INSTANCES;

    private static String _VERSION;

    private static final boolean _GROOVY_AVAILABLE;
    private static final boolean _OPENCL_AVAILABLE;

    static
    {
        _INSTANCES = new ThreadLocal<>();
        _GROOVY_AVAILABLE = Utility.isPresent( "groovy.lang.GroovySystem" );
        _OPENCL_AVAILABLE = Utility.isPresent( "org.jocl.CL" );
    }

    private final Settings _settings;
    private final Utility _utility;

    private Neureka(){
        _settings = new Settings();
        _utility = new Utility();
    }

    public static Neureka instance(){
        Neureka n = _INSTANCES.get();
        if( n == null ) {
            n = new Neureka();
            synchronized ( Neureka.class ) {
                setContext( n );
                n.reset(); // Initial reset must be synchronized because of dependency issues!
            }
        }
        return n;
    }

    public static void setContext( Neureka instance ) {
        _INSTANCES.set(instance);
    }

    public static Neureka instance(Object closure) {
        if(_GROOVY_AVAILABLE) {
            Object o = SettingsLoader.tryGroovyClosureOn(closure, Neureka.instance());
            if (o instanceof String) _VERSION = (String) o;
        }
        return Neureka.instance();
    }

    public boolean canAccessGroovy(){
        return _GROOVY_AVAILABLE;
    }

    public boolean canAccessOpenCL(){
        return _OPENCL_AVAILABLE;
    }

    public Settings settings(){
        return _settings;
    }

    public Settings settings(Object closure){
        if(_GROOVY_AVAILABLE) SettingsLoader.tryGroovyClosureOn(closure, _settings);
        return _settings;
    }

    public Utility utility(){
        return _utility;
    }

    public static String version(){
        return _VERSION;
    }

    public void reset() {
        if (_GROOVY_AVAILABLE) {
            SettingsLoader.tryGroovyScriptsOn(this);
        } else {
            settings().autograd().setIsRetainingPendingErrorForJITProp( true );
            settings().autograd().setIsApplyingGradientWhenTensorIsUsed( true );
            settings().autograd().setIsApplyingGradientWhenRequested( true );
            settings().indexing().setIsUsingLegacyIndexing( false );
            settings().indexing().setIsUsingArrayBasedIndexing( true );
            settings().debug().setIsKeepingDerivativeTargetPayloads( false );
            settings().view().setIsUsingLegacyView( false );
        }
    }

    private boolean _currentThreadIsAuthorized() {
        return this.equals( _INSTANCES.get() );
    }

    public class Settings
    {
        private final Debug _debug;
        private final AutoGrad _autograd;
        private final Indexing _indexing;
        private final View _view;
        private final NDim _ndim;
        private final DType _dtype;

        private boolean _isLocked = false;

        private Settings() {
            _debug = new Debug();
            _autograd = new AutoGrad();
            _indexing = new Indexing();
            _view = new View();
            _ndim = new NDim();
            _dtype = new DType();
        }

        public Debug debug() {
            return _debug;
        }

        public Debug debug(Object closure) {
            if(_GROOVY_AVAILABLE) SettingsLoader.tryGroovyClosureOn(closure, _debug);
            return _debug;
        }

        public AutoGrad autograd(){
            return _autograd;
        }

        public AutoGrad autograd( Object closure ) {
            if(_GROOVY_AVAILABLE) SettingsLoader.tryGroovyClosureOn( closure, _autograd );
            return _autograd;
        }

        public Indexing indexing(){
            return _indexing;
        }

        public Indexing indexing( Object closure ) {
            if(_GROOVY_AVAILABLE) SettingsLoader.tryGroovyClosureOn( closure, _indexing );
            return _indexing;
        }

        public View view(){
            return _view;
        }

        public View view( Object closure ) {
            if(_GROOVY_AVAILABLE) SettingsLoader.tryGroovyClosureOn( closure, _view );
            return _view;
        }

        public NDim ndim(){
            return _ndim;
        }

        public NDim ndim( Object closure ) {
            if(_GROOVY_AVAILABLE) SettingsLoader.tryGroovyClosureOn( closure, _ndim );
            return _ndim;
        }

        public DType dtype() {
            return _dtype;
        }

        public DType dtype( Object closure ) {
            if(_GROOVY_AVAILABLE) SettingsLoader.tryGroovyClosureOn( closure, _dtype );
            return _dtype;
        }

        public boolean isLocked(){
            return  _isLocked;
        }

        public void setIsLocked(boolean locked) {
            _isLocked = locked;
        }

        public class Debug
        {
            /**
             * Every derivative is calculated with respect to some graph node.
             * Graph nodes contain payload tensors.
             * A tensor might not always be used for backpropagation.
             * Therefore it will be deleted if possible.
             * Targeted tensors are either leave tensors (They require gradients)
             * or they are angle points between forward- and reverse-mode-AutoDiff!
             * In this case:
             * If the tensor is not needed for backpropagation it will be deleted.
             * The graph node will dereference the tensor either way.
             *
             * The flag determines this behavior with respect to target nodes.
             * It is used in the test suit to validate that the right tensors were calculated.
             * This flag should not be modified in production! (memory leak)
             */
            private boolean _isKeepingDerivativeTargetPayloads = false;

            public boolean isKeepingDerivativeTargetPayloads(){
                return _isKeepingDerivativeTargetPayloads;
            }

            public void setIsKeepingDerivativeTargetPayloads(boolean keep){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isKeepingDerivativeTargetPayloads = keep;
            }

        }

        public class AutoGrad // Auto-Grad/Differentiation
        {

            private boolean _isPreventingInlineOperations = true;

            /**
             * This flag enables an optimization technique which only propagates error values to
             * gradients if needed by a tensor (the tensor is used again) and otherwise accumulate them
             * at divergent differentiation paths within the computation graph.<br>
             * If the flag is set to true <br>
             * then error values will accumulate at such junction nodes.
             * This technique however uses more memory but will
             * improve performance for some networks substantially.
             * The technique is termed JIT-Propagation.
             */
            private boolean _isRetainingPendingErrorForJITProp = true;

            /**
             * Gradients will automatically be applied (or JITed) to tensors as soon as
             * they are being used for calculation (GraphNode instantiation).
             * This feature works well with JIT-Propagation.
             */
            private boolean _isApplyingGradientWhenTensorIsUsed = true;

            /**
             * Gradients will only be applied if requested.
             * Usually this happens immediately, however
             * if the flag 'applyGradientWhenTensorIsUsed' is set
             * to true, then the tensor will only be updated by its
             * gradient if requested AND the tensor is used fo calculation! (GraphNode instantiation).
             */
            private boolean _isApplyingGradientWhenRequested = true;

            public boolean isPreventingInlineOperations(){
                return _isPreventingInlineOperations;
            }

            public void setIsPreventingInlineOperations(boolean prevent) {
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isPreventingInlineOperations = prevent;
            }

            public boolean isRetainingPendingErrorForJITProp(){
                return _isRetainingPendingErrorForJITProp;
            }

            public void setIsRetainingPendingErrorForJITProp(boolean retain){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isRetainingPendingErrorForJITProp = retain;
            }

            public boolean isApplyingGradientWhenTensorIsUsed(){
                return _isApplyingGradientWhenTensorIsUsed;
            }

            public void setIsApplyingGradientWhenTensorIsUsed(boolean apply){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isApplyingGradientWhenTensorIsUsed = apply;
            }

            public boolean isApplyingGradientWhenRequested(){
                return _isApplyingGradientWhenRequested;
            }

            public void setIsApplyingGradientWhenRequested(boolean apply) {
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isApplyingGradientWhenRequested = apply;
            }

        }

        public class Indexing
        {
            private boolean _isUsingLegacyIndexing = false;

            private boolean _isUsingArrayBasedIndexing = true;



            public boolean isUsingLegacyIndexing(){
                return _isUsingLegacyIndexing;
            }

            public void setIsUsingLegacyIndexing(boolean enabled) {
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isUsingLegacyIndexing = enabled; // NOTE: gpu code must recompiled! (in OpenCLPlatform)
            }

            public boolean isUsingArrayBasedIndexing(){
                return _isUsingArrayBasedIndexing;
            }

            public void setIsUsingArrayBasedIndexing( boolean thorough ) {
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isUsingArrayBasedIndexing = thorough;
            }

        }

        public class View
        {
            private boolean _isUsingLegacyView = false;

            public boolean isUsingLegacyView(){
                return _isUsingLegacyView;
            }

            public void setIsUsingLegacyView(boolean enabled){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isUsingLegacyView = enabled;
            }

        }

        public class NDim
        {
            /**
             *  The DefaultNDConfiguration class stores shape, translation...
             *  as cached int arrays.
             *  Disabling this flag allows for custom 1D, 2D, 3D classes to be loaded. (Improves memory locality)
             */
            private boolean _isOnlyUsingDefaultNDConfiguration = false;

            public boolean isOnlyUsingDefaultNDConfiguration(){
                return _isOnlyUsingDefaultNDConfiguration;
            }

            public void setIsOnlyUsingDefaultNDConfiguration(boolean enabled){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isOnlyUsingDefaultNDConfiguration = enabled;
            }

        }

        public class DType {

            private Class<?> _defaultDataTypeClass = F64.class;

            public Class getDefaultDataTypeClass(){
                return _defaultDataTypeClass;
            }

            public void setDefaultDataTypeClass( Class dtype ){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _defaultDataTypeClass = dtype;
            }

        }

    }

    public static class Utility {
        /**
         * Helper method which reads the file with the given name and returns
         * the contents of this file as a String. Will exit the application
         * if the file can not be read.
         *
         * @param path
         * @return The contents of the file
         */
        public String readResource(String path){
            InputStream stream = getClass().getClassLoader().getResourceAsStream(path);
            try {
                BufferedReader br = new BufferedReader(new InputStreamReader(stream));
                StringBuffer sb = new StringBuffer();
                String line = "";
                while (line!=null) {
                    line = br.readLine();
                    if (line != null) sb.append(line).append("\n");
                }
                return sb.toString();
            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
                return null;
            }
        }

        public static boolean isPresent(String className){
            boolean found = false;
            String groovyInfo = ((className.toLowerCase().contains("groovy"))?" Neureka settings uninitialized!":"");
            String cause = " unknown ";
            try {
                Class.forName(className);
                found = true;
            } catch (Throwable ex) {// Class or one of its dependencies is not present...
                cause = ex.getMessage();
            } finally {
                if(!found){
                    System.out.println(
                            "[Info]: '"+className+"' dependencies not found!"+groovyInfo+"\n[Cause]: "+cause
                    );
                }
                return found;
            }
        }

    }


}
