import qupath.lib.scripting.QP
import qupath.lib.geom.Point2
import qupath.lib.roi.PolygonROI
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.images.servers.ImageServer
import qupath.lib.gui.dialogs.Dialogs     // new in v0.2.0-m10

//Aperio Image Scope displays images in a different orientation
def rotated = false

def server = getCurrentImageData().getServer()
def h = server.getHeight()
def w = server.getWidth()

// need to add annotations to hierarchy so qupath sees them
def hierarchy = getCurrentHierarchy()

//Prompt user for exported aperio image scope annotation file
//def file = getQuPath().getDialogHelper().promptForFile('xml', null, 'aperio xml file', null)
def file = Dialogs.promptForFile('xml', null, 'aperio xml file', null)     // new in v0.2.0-m10
def text = file.getText()

def list = new XmlSlurper().parseText(text)

list.Annotation.each {

    it.Regions.Region.each { region ->

        def tmp_points_list = []

        region.Vertices.Vertex.each{ vertex ->

            if (rotated) {
                X = vertex.@Y.toDouble()
                Y = h - vertex.@X.toDouble()
            }
            else {
                X = vertex.@X.toDouble()
                Y = vertex.@Y.toDouble()
            }
            tmp_points_list.add(new Point2(X, Y))
        }

        def roi = new PolygonROI(tmp_points_list)

        def annotation = new PathAnnotationObject(roi)

        hierarchy.addPathObject(annotation, true)
    }
}
print 'Done!'
