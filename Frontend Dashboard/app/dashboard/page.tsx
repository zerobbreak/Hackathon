"use client"

import { Bell, MapPin, Activity, AlertTriangle, MessageSquare, Users, TrendingUp, Clock } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

export default function DashboardPage() {
  const handleViewDetails = (alertId: string) => {
    alert(`Viewing details for alert: ${alertId}`);
    // In a real app, this would navigate to the alert details page or open a modal
  };

  const handleSendMassAlert = () => {
    alert("Sending mass alert to all affected communities!");
  };

  const handleDeployTeam = () => {
    alert("Initiating team deployment process!");
  };

  const handleUpdateAreaStatus = () => {
    alert("Opening form to update affected area status!");
  };

  return (
    <>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {/* Crisis Summary Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Crises</CardTitle>
            <AlertTriangle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">3</div>
            <p className="text-xs text-muted-foreground">Currently being managed</p>
            <Progress value={60} className="mt-2 h-2" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Affected Communities</CardTitle>
            <MapPin className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">Under flood warning</p>
            <Progress value={80} className="mt-2 h-2" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Alerts Sent (24h)</CardTitle>
            <Bell className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">87</div>
            <p className="text-xs text-muted-foreground">Total notifications</p>
            <Progress value={95} className="mt-2 h-2" />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        {/* Alerts Trend Chart Card */}
        <Card className="lg:col-span-4">
          <CardHeader>
            <CardTitle>Alerts Trend (Last 7 Days)</CardTitle>
            <CardDescription>Visualizing the volume of alerts over time.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full bg-gray-100 flex items-center justify-center rounded-lg text-muted-foreground border border-dashed">
              <TrendingUp className="h-8 w-8 mr-2" /> Placeholder for Chart (e.g., Line Chart)
            </div>
          </CardContent>
        </Card>

        {/* Recent Alerts Card - with more detail */}
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle>Recent Flood Alerts</CardTitle>
            <CardDescription>Latest alerts issued by CrisisConnect with quick actions.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="flex items-center gap-4">
              <div className="grid gap-1">
                <p className="text-sm font-medium leading-none">Urgent Flood Warning - Riverbend</p>
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Clock className="h-3 w-3" /> 5 mins ago &middot; Severity: High
                </p>
              </div>
              <Badge variant="destructive" className="ml-auto">
                Urgent
              </Badge>
              <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => handleViewDetails("A001")}>
                <Activity className="h-4 w-4" />
                <span className="sr-only">View Details</span>
              </Button>
            </div>
            <div className="flex items-center gap-4">
              <div className="grid gap-1">
                <p className="text-sm font-medium leading-none">Advisory - Coastal Area A</p>
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Clock className="h-3 w-3" /> 30 mins ago &middot; Severity: Medium
                </p>
              </div>
              <Badge variant="destructive" className="ml-auto">
                Advisory
              </Badge>
              <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => handleViewDetails("A002")}>
                <Activity className="h-4 w-4" />
                <span className="sr-only">View Details</span>
              </Button>
            </div>
            <div className="flex items-center gap-4">
              <div className="grid gap-1">
                <p className="text-sm font-medium leading-none">Flood Watch - Lowlands Sector 7</p>
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Clock className="h-3 w-3" /> 1 hour ago &middot; Severity: Low
                </p>
              </div>
              <Badge variant="outline" className="ml-auto">
                Watch
              </Badge>
              <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => handleViewDetails("A003")}>
                <Activity className="h-4 w-4" />
                <span className="sr-only">View Details</span>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions & System Status */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>Perform common crisis management tasks instantly.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-2">
            <Button className="justify-start bg-blue-500 hover:bg-blue-600 text-white" onClick={handleSendMassAlert}>
              <MessageSquare className="h-4 w-4 mr-2" /> Send Mass Alert
            </Button>
            <Button variant="outline" className="justify-start bg-transparent" onClick={handleDeployTeam}>
              <Users className="h-4 w-4 mr-2" /> Deploy Response Team
            </Button>
            <Button variant="outline" className="justify-start bg-transparent" onClick={handleUpdateAreaStatus}>
              <MapPin className="h-4 w-4 mr-2" /> Update Affected Area Status
            </Button>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>System Health & Connectivity</CardTitle>
            <CardDescription>Real-time overview of CrisisConnect system status.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Data Feeds</span>
              <Badge className="bg-green-500 hover:bg-green-600 text-white">Operational</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Alert Delivery Network</span>
              <Badge className="bg-green-500 hover:bg-green-600 text-white">Operational</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Local Infrastructure Links</span>
              <Badge variant="destructive">Degraded (3 areas)</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Database Connection</span>
              <Badge className="bg-green-500 hover:bg-green-600 text-white">Operational</Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  )
}
