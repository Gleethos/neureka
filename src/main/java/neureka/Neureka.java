package neureka;

import groovy.lang.Closure;
import groovy.lang.GroovyShell;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;


public class Neureka
{
    private static final Neureka _instance;
    private Settings _settings;
    private Utility _utility;

    static {
        _instance = new Neureka();
        new GroovyShell().evaluate(_instance.utility().readResource("library_settings.groovy"));
        new GroovyShell().evaluate(_instance.utility().readResource("scripting_setup.groovy"));
    }

    private Neureka(){
            _settings = new Settings();
            _utility = new Utility();
    }

    public static Neureka instance(){
        return _instance;
    }

    public static Neureka instance(Closure c){
        c.setDelegate(_instance);
        c.call();
        return _instance;
    }

    public Settings settings(){
        return _settings;
    }

    public Settings settings(Closure c){
        c.setDelegate(_settings);
        c.call();
        return _settings;
    }

    public Utility utility(){
        return _utility;
    }

    public static String version(){
        return "1.0.0";
    }

    public class Settings
    {
        private Debug _debug;
        private AutoDiff _autoDiff;
        private Indexing _indexing;
        private View _view;

        private boolean _isLocked = false;

        private Settings(){
            _debug = new Debug();
            _autoDiff = new AutoDiff();
            _indexing = new Indexing();
            _view = new View();
        }

        public Debug debug(){
            return _debug;
        }

        public Debug debug(Closure c){
            c.setDelegate(_debug);
            c.call();
            return _debug;
        }

        public AutoDiff autoDiff(){
            return _autoDiff;
        }

        public AutoDiff autoDiff(Closure c){
            c.setDelegate(_autoDiff);
            c.call();
            return _autoDiff;
        }

        public Indexing indexing(){
            return _indexing;
        }

        public Indexing indexing(Closure c){
            c.setDelegate(_indexing);
            c.call();
            return _indexing;
        }

        public View view(){
            return _view;
        }

        public View view(Closure c){
            c.setDelegate(_view);
            c.call();
            return _view;
        }

        public boolean isLocked(){
            return  _isLocked;
        }

        public void setIsLocked(boolean locked){
            _isLocked = locked;
        }

        public void reset(){
            debug().reset();
            autoDiff().reset();
            indexing().reset();
            view().reset();
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
            private boolean _keepDerivativeTargetPayloads;

            Debug(){
                reset();
            }

            public void reset(){
                _keepDerivativeTargetPayloads = false;
            }

            public boolean keepDerivativeTargetPayloads(){
                return _keepDerivativeTargetPayloads;
            }

            public void setKeepDerivativeTargetPayloads(boolean keep){
                if(_isLocked) return;
                _keepDerivativeTargetPayloads = keep;
            }

        }

        public class AutoDiff // Auto-Differentiation
        {
            /**
             * This flag enables an optimization technique which only applies
             * gradients as soon as they are needed by a tensor (the tensor is used again).
             * If the flag is set to true
             * then error values will accumulate whenever it makes sense.
             * This technique however uses more memory but will
             * improve performance for some networks substantially.
             * The technique is termed JIT-Propagation.
             */
            private boolean _retainPendingErrorForJITProp;

            /**
             * Gradients will automatically be applied to tensors as soon as
             * they are being used for calculation.
             * This feature works well with JIT-Propagation.
             */
            private boolean _applyGradientWhenTensorIsUsed;

            AutoDiff(){
                reset();
            }

            public void reset(){
                _retainPendingErrorForJITProp = true;
                _applyGradientWhenTensorIsUsed = false;
            }

            public boolean retainPendingErrorForJITProp(){
                return _retainPendingErrorForJITProp;
            }

            public void setRetainPendingErrorForJITProp(boolean retain){
                if(_isLocked) return;
                _retainPendingErrorForJITProp = retain;
            }

            public boolean applyGradientWhenTensorIsUsed(){
                return _applyGradientWhenTensorIsUsed;
            }

            public void setApplyGradientWhenTensorIsUsed(boolean apply){
                if(_isLocked) return;
                _applyGradientWhenTensorIsUsed = apply;
            }

        }

        public class Indexing
        {
            private boolean _legacyIndexing;

            Indexing(){
                reset();
            }

            public void reset(){
                _legacyIndexing = false;
            }

            public boolean legacy(){
                return _legacyIndexing;
            }

            public void setLegacy(boolean enabled){
                if(_isLocked) return;
                _legacyIndexing = enabled;//NOTE: gpu code must recompiled! (in OpenCLPlatform)
            }

        }

        public class View
        {

            private boolean _legacyView;

            View() {
                reset();
            }

            public void reset(){
                _legacyView = false;
            }

            public boolean legacy(){
                return _legacyView;
            }

            public void setLegacy(boolean enabled){
                if(_isLocked) return;
                _legacyView = enabled;
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
            InputStream stream = _instance.getClass().getClassLoader().getResourceAsStream(path);
            try {
                BufferedReader br = new BufferedReader(new InputStreamReader(stream));//new FileInputStream(fileName)));
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





    }


}
