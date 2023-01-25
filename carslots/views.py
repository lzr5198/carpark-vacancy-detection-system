from django.shortcuts import render
from django.http import HttpResponse
from .models import Carslot

# Create your views here.
def test(request):
    # field_name = 'slotId'
    # obj = Carslot.objects.first()
    # field_object = Carslot._meta.get_field(field_name)
    # field_value = field_object.value_from_object(obj)
    if request.method == "POST":
        object_name = request.POST["object_name"]
        x1 = request.POST["x1"]
        y1 = request.POST["y1"]
        x2 = request.POST["x2"]
        y2 = request.POST["y2"]

        print(object_name)
        print(x1)
        print(y1)
        print(x2)
        print(y2)
    else:
        print("Post not received")

    return HttpResponse("<h1>test</h1>")