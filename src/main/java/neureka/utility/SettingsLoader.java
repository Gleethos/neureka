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

    _____      _   _   _                 _                     _
   / ____|    | | | | (_)               | |                   | |
  | (___   ___| |_| |_ _ _ __   __ _ ___| |     ___   __ _  __| | ___ _ __
   \___ \ / _ \ __| __| | '_ \ / _` / __| |    / _ \ / _` |/ _` |/ _ \ '__|
   ____) |  __/ |_| |_| | | | | (_| \__ \ |___| (_) | (_| | (_| |  __/ |
  |_____/ \___|\__|\__|_|_| |_|\__, |___/______\___/ \__,_|\__,_|\___|_|
                                __/ |
                               |___/

    A simply utility class used by the Neureka singleton instance for configuration loading...
*/

package neureka.utility;

import groovy.lang.Closure;
import groovy.lang.GroovyShell;
import groovy.lang.GroovySystem;
import neureka.Neureka;

/**
 *  This class is a helper class for Neureka instances.
 *  It tries to execute groovy scripts used as settings for said instances.
 *
 *  This logic is not included inside the Neureka class
 *  itself because otherwise there would be an obligate dependency
 *  on groovy.
 *  If groovy lang dependencies are however not found, then
 *  this very class will not be used, initialized and therefore
 *  neureka will continue to work without groovy.
 */
public class SettingsLoader
{
    private static String _settings_source;
    private static String _setup_source;

    public static Object tryGroovyClosureOn(Object closure, Object delegate) {
            ( (Closure) closure ).setDelegate(delegate);
            return ( (Closure) closure ).call(delegate);
    }

    public static void tryGroovyScriptsOn(Neureka instance)
    {
        if ( _settings_source == null || _setup_source == null) {
            _settings_source = instance.utility().readResource("library_settings.groovy");
            _setup_source = instance.utility().readResource("scripting_setup.groovy");
        }
        try {

            String version = GroovySystem.getVersion();
            if (Integer.parseInt(version.split("\\.")[ 0 ]) < 3) {
                throw new IllegalStateException(
                        "Wrong groovy version "+version+" found! Version 3.0.0 or greater required."
                );
            }
            new GroovyShell(instance.getClass().getClassLoader()).evaluate(_settings_source);
            new GroovyShell(instance.getClass().getClassLoader()).evaluate(_setup_source);

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
