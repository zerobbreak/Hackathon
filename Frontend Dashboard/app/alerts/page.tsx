"use client"

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Plus, Search, Filter, Eye, Edit, Trash2, Activity, CheckSquare, Archive, Clock } from 'lucide-react'
import { Input } from "@/components/ui/input"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"

export default function AlertsPage() {
  const initialAlerts = [
    {
      id: "A001",
      type: "Flood Warning",
      location: "Riverbend",
      severity: "High",
      status: "Active",
      issued: "2025-08-05 10:00 AM",
      details: "River levels rising rapidly. Evacuation advisory for low-lying areas.",
    },
    {
      id: "A002",
      type: "Advisory",
      location: "Coastal Area A",
      severity: "Medium",
      status: "Active",
      issued: "2025-08-05 09:30 AM",
      details: "Strong currents expected near the coast. Avoid recreational water activities.",
    },
    {
      id: "A003",
      type: "Flood Watch",
      location: "Lowlands Sector 7",
      severity: "Low",
      status: "Active",
      issued: "2025-08-05 08:00 AM",
      details: "Heavy rainfall predicted. Monitor local conditions for potential flooding.",
    },
    {
      id: "A004",
      type: "Evacuation Order",
      location: "Hillside Village",
      severity: "Critical",
      status: "Active",
      issued: "2025-08-04 06:00 PM",
      details: "Mandatory evacuation due to landslide risk. Proceed to designated shelters.",
    },
    {
      id: "A005",
      type: "Warning",
      location: "Central Plains",
      severity: "Medium",
      status: "Resolved",
      issued: "2025-08-03 02:00 PM",
      details: "Flash flood warning has been lifted. Roads are now clear.",
    },
  ];

  const [alerts, setAlerts] = useState(initialAlerts);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedAlerts, setSelectedAlerts] = useState<string[]>([]);

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case "High":
      case "Critical":
        return <Badge variant="destructive">{severity}</Badge>
      case "Medium":
        return <Badge variant="secondary">{severity}</Badge>
      case "Low":
        return <Badge variant="outline">{severity}</Badge>
      default:
        return <Badge>{severity}</Badge>
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "Active":
        return <Badge className="bg-green-500 hover:bg-green-600 text-white">{status}</Badge>
      case "Resolved":
        return <Badge variant="secondary">{status}</Badge>
      default:
        return <Badge>{status}</Badge>
    }
  }

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
    // Implement actual filtering logic here if needed
    console.log("Searching for:", e.target.value);
  };

  const handleFilter = (filterType: string, value: string) => {
    alert(`Filtering by ${filterType}: ${value}`);
    // Implement actual filtering logic here
  };

  const handleNewAlert = () => {
    alert("Opening form to create a new alert!");
  };

  const handleEditAlert = (id: string) => {
    alert(`Editing alert: ${id}`);
    // In a real app, this would navigate to an edit form or open a modal
  };

  const handleDeleteAlert = (id: string) => {
    if (confirm(`Are you sure you want to delete alert ${id}?`)) {
      setAlerts(alerts.filter(alert => alert.id !== id));
      alert(`Alert ${id} deleted.`);
    }
  };

  const handleSelectAlert = (id: string, checked: boolean) => {
    setSelectedAlerts(prev =>
      checked ? [...prev, id] : prev.filter(alertId => alertId !== id)
    );
  };

  const handleSelectAllAlerts = (checked: boolean) => {
    if (checked) {
      setSelectedAlerts(alerts.map(alert => alert.id));
    } else {
      setSelectedAlerts([]);
    }
  };

  const handleBatchAction = (action: string) => {
    if (selectedAlerts.length === 0) {
      alert("Please select at least one alert for batch action.");
      return;
    }
    if (confirm(`Are you sure you want to ${action} ${selectedAlerts.length} selected alerts?`)) {
      alert(`Performing batch action: ${action} on alerts: ${selectedAlerts.join(', ')}`);
      // Implement actual batch logic here (e.g., update status, delete from state)
      setSelectedAlerts([]); // Clear selection after action
    }
  };

  const filteredAlerts = alerts.filter(alert =>
    alert.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    alert.location.toLowerCase().includes(searchTerm.toLowerCase()) ||
    alert.type.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>All Alerts</CardTitle>
        <CardDescription>Manage and view all flood warnings and advisories.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-4 mb-4">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search alerts by ID, location, or type..."
              className="w-full rounded-lg bg-background pl-8"
              value={searchTerm}
              onChange={handleSearchChange}
            />
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="gap-1 bg-transparent">
                <Filter className="h-3.5 w-3.5" />
                <span className="sr-only sm:not-sr-only sm:whitespace-nowrap">Filter</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => handleFilter("status", "Active")}>Status: Active</DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleFilter("status", "Resolved")}>Status: Resolved</DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleFilter("severity", "High")}>Severity: High</DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleFilter("severity", "Critical")}>Severity: Critical</DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleFilter("location", "Riverbend")}>Location: Riverbend</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="gap-1 bg-transparent" disabled={selectedAlerts.length === 0}>
                <Activity className="h-3.5 w-3.5" />
                <span className="sr-only sm:not-sr-only sm:whitespace-nowrap">Batch Actions</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => handleBatchAction("mark as resolved")}>
                <CheckSquare className="mr-2 h-4 w-4" /> Mark Selected as Resolved
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleBatchAction("archive")}>
                <Archive className="mr-2 h-4 w-4" /> Archive Selected
              </DropdownMenuItem>
              <DropdownMenuItem className="text-red-600" onClick={() => handleBatchAction("delete")}>
                <Trash2 className="mr-2 h-4 w-4" /> Delete Selected
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <Button size="sm" className="h-8 gap-1" onClick={handleNewAlert}>
            <Plus className="h-3.5 w-3.5" />
            <span className="sr-only sm:not-sr-only sm:whitespace-nowrap">New Alert</span>
          </Button>
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[30px]">
                <Checkbox
                  aria-label="Select all"
                  checked={selectedAlerts.length === alerts.length && alerts.length > 0}
                  onCheckedChange={(checked: boolean) => handleSelectAllAlerts(checked)}
                />
              </TableHead>
              <TableHead>ID</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Location</TableHead>
              <TableHead>Severity</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Issued</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredAlerts.map((alert) => (
              <TableRow key={alert.id}>
                <TableCell>
                  <Checkbox
                    aria-label={`Select alert ${alert.id}`}
                    checked={selectedAlerts.includes(alert.id)}
                    onCheckedChange={(checked: boolean) => handleSelectAlert(alert.id, checked)}
                  />
                </TableCell>
                <TableCell className="font-medium">{alert.id}</TableCell>
                <TableCell>{alert.type}</TableCell>
                <TableCell>{alert.location}</TableCell>
                <TableCell>{getSeverityBadge(alert.severity)}</TableCell>
                <TableCell>{getStatusBadge(alert.status)}</TableCell>
                <TableCell className="flex items-center gap-1">
                  <Clock className="h-3 w-3 text-muted-foreground" /> {alert.issued}
                </TableCell>
                <TableCell className="text-right">
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button variant="ghost" size="icon">
                        <Eye className="h-4 w-4" />
                        <span className="sr-only">View Details</span>
                      </Button>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>Alert Details: {alert.id}</DialogTitle>
                        <DialogDescription>
                          Comprehensive information about this alert.
                        </DialogDescription>
                      </DialogHeader>
                      <div className="grid gap-4 py-4">
                        <div className="grid grid-cols-4 items-center gap-4">
                          <Label className="text-right">Type</Label>
                          <div className="col-span-3">{alert.type}</div>
                        </div>
                        <div className="grid grid-cols-4 items-center gap-4">
                          <Label className="text-right">Location</Label>
                          <div className="col-span-3">{alert.location}</div>
                        </div>
                        <div className="grid grid-cols-4 items-center gap-4">
                          <Label className="text-right">Severity</Label>
                          <div className="col-span-3">{getSeverityBadge(alert.severity)}</div>
                        </div>
                        <div className="grid grid-cols-4 items-center gap-4">
                          <Label className="text-right">Status</Label>
                          <div className="col-span-3">{getStatusBadge(alert.status)}</div>
                        </div>
                        <div className="grid grid-cols-4 items-center gap-4">
                          <Label className="text-right">Issued</Label>
                          <div className="col-span-3">{alert.issued}</div>
                        </div>
                        <div className="grid grid-cols-4 items-center gap-4">
                          <Label className="text-right">Details</Label>
                          <div className="col-span-3">{alert.details}</div>
                        </div>
                      </div>
                    </DialogContent>
                  </Dialog>
                  <Button variant="ghost" size="icon" onClick={() => handleEditAlert(alert.id)}>
                    <Edit className="h-4 w-4" />
                    <span className="sr-only">Edit</span>
                  </Button>
                  <Button variant="ghost" size="icon" className="text-red-600" onClick={() => handleDeleteAlert(alert.id)}>
                    <Trash2 className="h-4 w-4" />
                    <span className="sr-only">Delete</span>
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
